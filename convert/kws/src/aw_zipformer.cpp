#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>

#include "process.h"
#include "aw_zipformer.h"
#include "kws_decode.h"

#include <algorithm>
#include <cmath>
#include <memory>
#include <unordered_map>
#include <vector>

/* vip_buffer_format_e (keep numeric to avoid pulling vip_lite.h here) */
enum {
    kFmtFP32 = 0,
    kFmtFP16 = 1,
    kFmtINT16 = 5,
    kFmtINT32 = 8,
    kFmtINT64 = 10
};

static const char *fmt_name(int f)
{
    switch (f) {
    case kFmtFP32: return "FP32";
    case kFmtFP16: return "FP16";
    case kFmtINT16: return "INT16";
    case kFmtINT32: return "INT32";
    case kFmtINT64: return "INT64";
    default: return "?";
    }
}

/* Pack ONNX-dense src[rows][cols] into VIP WHCN (W innermost). */
static void pack_hw_to_vip(const float *src, int rows, int cols,
                           void *dst, unsigned dst_bytes, const tensor_desc_s &d)
{
    unsigned need = (unsigned)rows * (unsigned)cols * sizeof(float);
    unsigned ncopy = need < dst_bytes ? need : dst_bytes;
    unsigned W = d.sizes[0] ? d.sizes[0] : 1;
    unsigned H = (d.num_dims > 1 && d.sizes[1]) ? d.sizes[1] : 1;
    unsigned C = (d.num_dims > 2 && d.sizes[2]) ? d.sizes[2] : 1;
    if (d.data_format != kFmtFP32) {
        memcpy(dst, src, ncopy);
        return;
    }
    if (W == (unsigned)cols && H * C == (unsigned)rows) {
        memcpy(dst, src, ncopy);
        return;
    }
    /* Transposed: VIP W is ONNX row (time), H*C is ONNX col (feat). */
    if (W == (unsigned)rows && H * C == (unsigned)cols) {
        float *out = (float *)dst;
        memset(out, 0, dst_bytes);
        for (int r = 0; r < rows; ++r)
            for (int c = 0; c < cols; ++c)
                out[c * rows + r] = src[r * cols + c];
        return;
    }
    memcpy(dst, src, ncopy);
}

static void unpack_vip_to_hw(const float *src, float *dst, int rows, int cols,
                             const tensor_desc_s &d)
{
    unsigned W = d.sizes[0] ? d.sizes[0] : 1;
    unsigned H = (d.num_dims > 1 && d.sizes[1]) ? d.sizes[1] : 1;
    unsigned C = (d.num_dims > 2 && d.sizes[2]) ? d.sizes[2] : 1;
    if (d.data_format != kFmtFP32 ||
        (W == (unsigned)cols && H * C == (unsigned)rows)) {
        memcpy(dst, src, (size_t)rows * cols * sizeof(float));
        return;
    }
    if (W == (unsigned)rows && H * C == (unsigned)cols) {
        for (int r = 0; r < rows; ++r)
            for (int c = 0; c < cols; ++c)
                dst[r * cols + c] = src[c * rows + r];
        return;
    }
    memcpy(dst, src, (size_t)rows * cols * sizeof(float));
}

static int read_scalar_int(const void *p, int fmt)
{
    if (!p) return 0;
    switch (fmt) {
    case kFmtINT32: return *(const int32_t *)p;
    case kFmtINT64: return (int)(*(const int64_t *)p);
    case kFmtINT16: return *(const int16_t *)p;
    case kFmtFP32:  return (int)(*(const float *)p);
    default:        return (int)(*(const float *)p);
    }
}

static void write_scalar_int(void *p, unsigned nbytes, int fmt, int value)
{
    if (!p) return;
    if (nbytes) memset(p, 0, nbytes);
    switch (fmt) {
    case kFmtINT32: {
        int32_t v = value;
        memcpy(p, &v, sizeof(v));
        break;
    }
    case kFmtINT64: {
        int64_t v = value;
        memcpy(p, &v, sizeof(v));
        break;
    }
    case kFmtINT16: {
        int16_t v = (int16_t)value;
        memcpy(p, &v, sizeof(v));
        break;
    }
    case kFmtFP32:
    default: {
        float v = (float)value;
        memcpy(p, &v, sizeof(v));
        break;
    }
    }
}

Encoder::Encoder(const char* model_path)
{
    int status = 0;
    network_idx = 0;    // first network
    plens_index = -1;
    memset(cache_bytes, 0, sizeof(cache_bytes));

    status = net.network_create((char*)model_path, network_idx);
    if (status != 0) {
        printf("Failed to create network, status=%d, network_idx=%d.\n", status, network_idx);
        return ;
    }

    status = net.network_prepare();
    if (status != 0) {
        printf("Failed to prepare network, status=%d, network_idx=%d.\n", status, network_idx);
        return ;
    }

    int n_in = net.get_input_cnt();
    int n_out = net.get_output_cnt();
    for (int i = 0; i < n_in; ++i) {
        if (strstr(net.m_input_desc[i].name, "processed_lens")) {
            plens_index = i;
            break;
        }
    }
    if (plens_index < 0 && n_in > 0)
        plens_index = n_in - 1;

    printf("encoder IO in=%d out=%d plens_idx=%d fmt=%s dim=%u\n",
           n_in, n_out, plens_index,
           plens_index >= 0 ? fmt_name(net.m_input_desc[plens_index].data_format) : "?",
           plens_index >= 0 ? net.m_input_desc[plens_index].sizes[0] : 0);
    const tensor_desc_s &xin = net.m_input_desc[0];
    const tensor_desc_s &xout = net.m_output_desc[0];
    printf("  x VIP W,H,C,N=%u,%u,%u,%u fmt=%s | encoder_out W,H,C,N=%u,%u,%u,%u fmt=%s\n",
           xin.sizes[0], xin.sizes[1], xin.sizes[2], xin.sizes[3], fmt_name(xin.data_format),
           xout.sizes[0], xout.sizes[1], xout.sizes[2], xout.sizes[3], fmt_name(xout.data_format));
    /* ONNX: x(1,29,80) enc(1,4,320) conv(1,128,7) key0(64,1,128) embed(1,128,3,19)
     * VIP WHCN W-innermost matches those last dims, so 回灌 is native memcpy. */
    if (n_in > 37) {
        const tensor_desc_s &k0 = net.m_input_desc[1];
        const tensor_desc_s &c0 = net.m_input_desc[5];
        const tensor_desc_s &em = net.m_input_desc[37];
        printf("  key0 %u,%u,%u conv1_0 %u,%u,%u embed %u,%u,%u,%u (VIP==ONNX last-dim, no transpose)\n",
               k0.sizes[0], k0.sizes[1], k0.sizes[2],
               c0.sizes[0], c0.sizes[1], c0.sizes[2],
               em.sizes[0], em.sizes[1], em.sizes[2], em.sizes[3]);
    }
    /* Flag caches whose VIP inner dim != ONNX last dim (need transpose if we
     * ever stored ONNX-dense). 回灌 itself stays VIP-native. */
    for (int i = 1; i < n_in && i < n_out; ++i) {
        const tensor_desc_s &in = net.m_input_desc[i];
        const tensor_desc_s &out = net.m_output_desc[i];
        if (in.sizes[0] != out.sizes[0] || in.n_elem != out.n_elem ||
            in.data_format != out.data_format) {
            printf("  cache[%d] IN/OUT mismatch %s %u,%u,%u,%u n=%u %s -> %s %u,%u,%u,%u n=%u %s\n",
                   i, in.name, in.sizes[0], in.sizes[1], in.sizes[2], in.sizes[3],
                   in.n_elem, fmt_name(in.data_format),
                   out.name, out.sizes[0], out.sizes[1], out.sizes[2], out.sizes[3],
                   out.n_elem, fmt_name(out.data_format));
        }
    }
}

Encoder::~Encoder()
{
    net.network_finish();
    net.network_destroy();
    printf("~Encoder. \n");
}

int Encoder::init_encoder_input(float ***input_data_ptr)
{
    if (input_data_ptr == nullptr) {
        printf("Error, input_data_ptr is nullptr!\n");
        return -1;
    }

    // get the number of model inputs
    int input_cnt = net.get_input_cnt();
    float** input_data = nullptr;

    try {
        input_data = new float*[input_cnt]();
        for (int i = 0; i < input_cnt; i++) {
            input_data[i] = nullptr;
        }

        for (int i = 0; i < input_cnt; i++) {
            unsigned bsz = 0;
            void *ptr = nullptr;
            net.get_network_input_buff_info(i, &ptr, &bsz);
            if (bsz < sizeof(float)) {
                int len = net.m_input_data_len[i];
                bsz = (unsigned)len * sizeof(float);
            }
            cache_bytes[i] = bsz;
            int nfloat = (int)((bsz + sizeof(float) - 1) / sizeof(float));
            input_data[i] = new float[nfloat];
            memset(input_data[i], 0, nfloat * sizeof(float));
        }

        // Return the allocated memory through parameters
        *input_data_ptr = input_data;
        return 0;

    } catch (const std::bad_alloc& e) {
        if (input_data != nullptr) {
            for (int i = 0; i < input_cnt; i++) {
                if (input_data[i] != nullptr) {
                    delete[] input_data[i];
                }
            }
            delete[] input_data;
        }
        printf("Memory allocation failed: %s\n", e.what());
        return -2;    // Memory allocation failed
    } catch (const std::exception& e) {
        if (input_data != nullptr) {
            for (int i = 0; i < input_cnt; i++) {
                if (input_data[i] != nullptr) {
                    delete[] input_data[i];
                }
            }
            delete[] input_data;
        }
        printf("Exception: %s\n", e.what());
        return -3;    // Other exceptions
    }
}

void Encoder::write_processed_lens(float **input_data, int value)
{
    if (!input_data || plens_index < 0) return;
    const tensor_desc_s &d = net.m_input_desc[plens_index];
    unsigned nbytes = cache_bytes[plens_index];
    if (nbytes == 0)
        nbytes = d.elem_bytes ? d.elem_bytes : (unsigned)sizeof(float);
    write_scalar_int(input_data[plens_index], nbytes, d.data_format, value);
}

int Encoder::run_encoder_model(float *encoder_input, float *encoder_output, float ***encoder_init_data)
{
    int status = 0;
    unsigned int input_buffer_size = 0;
    void *buffer_ptr = nullptr;
    net.get_network_input_buff_info(0, &buffer_ptr, &input_buffer_size);
    pack_hw_to_vip(encoder_input, N_SEGMENT, N_MELS, buffer_ptr, input_buffer_size,
                   net.m_input_desc[0]);

    int input_cnt = net.get_input_cnt();
    int nio = 0;
    void* buffer_ptrs[MAX_ENCODER_IO] = {nullptr};
    for (int i=1; i<input_cnt; i++) {
        unsigned int cache_buffer_size = 0;
        net.get_network_input_buff_info(i, &buffer_ptrs[i], &cache_buffer_size);
        unsigned ncopy = cache_bytes[i] ? cache_bytes[i] : cache_buffer_size;
        if (ncopy > cache_buffer_size && cache_buffer_size) ncopy = cache_buffer_size;
        memcpy(buffer_ptrs[i], (*encoder_init_data)[i], ncopy);
    }

    // network output count
    int output_cnt = net.get_output_cnt();
    float **output_data = new float*[output_cnt]();

    static int dump_n = 0;
    if (dump_n < 3 && getenv("DUMP_CACHE") && getenv("DUMP_CACHE")[0] == '1' &&
        plens_index >= 0) {
        int v = read_scalar_int((*encoder_init_data)[plens_index],
                                net.m_input_desc[plens_index].data_format);
        printf("ALIGN write processed_lens=%d fmt=%s (chunk#%d)\n",
               v, fmt_name(net.m_input_desc[plens_index].data_format), dump_n);
        dump_n++;
    }

    status = net.network_input_output_set();
    if (status != 0) {
        printf("Failed to set input/output, status=%d, network_idx=%d\n", status, network_idx);
        goto out;
    }

    status = net.network_run();
    if (status != 0) {
        printf("Failed to run network, status=%d, network_idx=%d\n", status, network_idx);
        goto out;
    }

    net.get_output(output_data);

    /* encoder_out: VIP WHCN → ONNX (T,C) for joiner. */
    unpack_vip_to_hw(output_data[0], encoder_output, ENCODER_OUTPUT_T, DECODER_DIM,
                     net.m_output_desc[0]);

    /* caches: copy VIP raw buffers including alignment padding so the next
     * chunk sees the same layout (回灌). Do NOT go through get_output float
     * conversion — processed_lens is INT and conv/key may have padding. */
    nio = net.get_input_cnt();
    if (output_cnt < nio) nio = output_cnt;
    for (int i = 1; i < nio; i++) {
        void *optr = nullptr;
        unsigned osz = 0, isz = cache_bytes[i];
        net.get_network_output_buff_info(i, &optr, &osz);
        unsigned ncopy = osz;
        if (isz && ncopy > isz) ncopy = isz;
        if (ncopy == 0) {
            ncopy = net.m_output_data_len[i] * sizeof(float);
            memcpy((*encoder_init_data)[i], output_data[i], ncopy);
        } else if (optr) {
            memcpy((*encoder_init_data)[i], optr, ncopy);
        }
        cache_bytes[i] = ncopy;
    }

    static int dump_once = 1;
    if (dump_once && getenv("DUMP_CACHE") && getenv("DUMP_CACHE")[0] == '1') {
        dump_once = 0;
        int last = plens_index >= 0 ? plens_index : (input_cnt - 1);
        void *pp = nullptr;
        unsigned psz = 0;
        net.get_network_output_buff_info(last, &pp, &psz);
        int new_plens = read_scalar_int(pp, net.m_output_desc[last].data_format);
        int in_plens = read_scalar_int((*encoder_init_data)[last],
                                       net.m_input_desc[last].data_format);
        printf("ALIGN plens in_fmt=%s out_fmt=%s new_processed_lens=%d (host_after_copy=%d)\n",
               fmt_name(net.m_input_desc[last].data_format),
               fmt_name(net.m_output_desc[last].data_format),
               new_plens, in_plens);
        /* conv caches: output 5,6,11,12,17,18,23,24,29,30,35,36 */
        int conv_idx[] = {5, 6, 11, 12, 17, 18, 23, 24, 29, 30, 35, 36};
        for (int ci = 0; ci < 12; ++ci) {
            int idx = conv_idx[ci];
            if (idx >= output_cnt || !output_data[idx]) continue;
            int n = (int)net.m_output_data_len[idx];
            int nzero = 0, firstz = -1, run0 = 0, maxrun = 0;
            int ch_alive = 0, ch_dead = 0;
            for (int k = 0; k < n; ++k) {
                int z = fabsf(output_data[idx][k]) < 1e-8f;
                if (z) {
                    nzero++;
                    if (firstz < 0) firstz = k;
                    run0++;
                    if (run0 > maxrun) maxrun = run0;
                } else {
                    run0 = 0;
                }
            }
            /* occupancy if laid out (C,7) with C=n/7 */
            if (n % 7 == 0) {
                int nc = n / 7;
                for (int ch = 0; ch < nc; ++ch) {
                    int dead = 1;
                    for (int t = 0; t < 7; ++t)
                        if (fabsf(output_data[idx][ch * 7 + t]) >= 1e-8f) { dead = 0; break; }
                    if (dead) ch_dead++; else ch_alive++;
                }
            }
            printf("npu out%d %s n=%d nzero=%d firstz=%d maxrun=%d ch_alive=%d ch_dead=%d first24:",
                   idx, net.m_output_desc[idx].name, n, nzero, firstz, maxrun, ch_alive, ch_dead);
            int show = n < 24 ? n : 24;
            for (int k = 0; k < show; ++k) printf(" %.4f", output_data[idx][k]);
            printf("\n");
            if (n % 7 == 0) {
                int nc = n / 7;
                printf("  ch_mask(C,7):");
                for (int ch = 0; ch < nc; ++ch) {
                    int dead = 1;
                    for (int t = 0; t < 7; ++t)
                        if (fabsf(output_data[idx][ch * 7 + t]) >= 1e-8f) { dead = 0; break; }
                    if (ch % 16 == 0) printf("\n   %3d:", ch);
                    printf("%c", dead ? '0' : '1');
                }
                printf("\n");
            }
            char fn[64];
            snprintf(fn, sizeof(fn), "/tmp/npu_out%d.bin", idx);
            FILE *fp = fopen(fn, "wb");
            if (fp) {
                fwrite(output_data[idx], sizeof(float), n, fp);
                fclose(fp);
            }
        }
        /* embed_states is typically output n-2 */
        int eidx = last - 1;
        if (eidx > 0 && output_data[eidx]) {
            printf("npu embed first8:");
            for (int k = 0; k < 8; ++k) printf(" %.5f", output_data[eidx][k]);
            printf("\n");
        }
        if (output_data[1]) {
            int nkey = (int)net.m_output_data_len[1];
            printf("npu key0 last8:");
            for (int k = nkey - 8; k < nkey; ++k) printf(" %.5f", output_data[1][k]);
            printf("\n");
        }
        printf("npu encoder_out first8:");
        for (int k = 0; k < 8; ++k) printf(" %.5f", encoder_output[k]);
        printf("\n");
    }

out:
    // Free dynamically allocated memory
    for (int i = 0; i < output_cnt; i++) {
        delete[] output_data[i];
        output_data[i] = nullptr;
    }
    delete[] output_data;
    output_data = nullptr;

    return status;
}


Decoder::Decoder(const char* model_path)
{
    int status = 0;
    network_idx = 1;    // second network

    status = net.network_create((char*)model_path, network_idx);
    if (status != 0) {
        printf("Failed to create network, status=%d, network_idx=%d.\n", status, network_idx);
        return ;
    }

    status = net.network_prepare();
    if (status != 0) {
        printf("Failed to prepare network, status=%d, network_idx=%d.\n", status, network_idx);
        return ;
    }
}

Decoder::~Decoder()
{
    net.network_finish();
    net.network_destroy();
    printf("~Decoder. \n");
}

int Decoder::run_decoder_model(float* decoder_input, float *decoder_output)
{
    int status = 0;
    void *input_buffer_ptr = nullptr;
    unsigned int input_buffer_size = 0;
    net.get_network_input_buff_info(0, &input_buffer_ptr, &input_buffer_size);
    memcpy(input_buffer_ptr, decoder_input, CONTEXT_SIZE * sizeof(float));

    // network output count
    int output_cnt = net.get_output_cnt();
    float **output_data = new float*[output_cnt]();


    status = net.network_input_output_set();
    if (status != 0) {
        printf("Failed to set input/output, status=%d, network_idx=%d\n", status, network_idx);
        goto out;
    }

    status = net.network_run();
    if (status != 0) {
        printf("Failed to run network, status=%d, network_idx=%d\n", status, network_idx);
        goto out;
    }

    net.get_output(output_data);

    if (output_cnt > 0 && output_data[0] != nullptr) {
        memcpy(decoder_output, output_data[0], net.m_output_data_len[0] * sizeof(float));
    }

out:
    for (int i = 0; i < output_cnt; i++) {
         delete[] output_data[i];
        output_data[i] = nullptr;
    }
    delete[] output_data;
    output_data = nullptr;

    return status;
}


Joiner::Joiner(const char* model_path)
{
    int status = 0;
    network_idx = 2;    // third network

    status = net.network_create((char*)model_path, network_idx);
    if (status != 0) {
        printf("Failed to create network, status=%d, network_idx=%d.\n", status, network_idx);
        return ;
    }

    status = net.network_prepare();
    if (status != 0) {
        printf("Failed to prepare network, status=%d, network_idx=%d.\n", status, network_idx);
        return ;
    }
}

Joiner::~Joiner()
{
    net.network_finish();
    net.network_destroy();
    printf("~Joiner. \n");
}

int Joiner::run_joiner_model(float* joiner_input0, float* joiner_input1, float *joiner_output)
{
    int status = 0;
    unsigned int input0_buffer_size = 0;
    unsigned int input1_buffer_size = 0;
    void *buffer_ptr_encoder = nullptr;
    void *buffer_ptr_decoder = nullptr;
    net.get_network_input_buff_info(0, &buffer_ptr_encoder, &input0_buffer_size);
    net.get_network_input_buff_info(1, &buffer_ptr_decoder, &input1_buffer_size);
    memcpy(buffer_ptr_encoder, joiner_input0, DECODER_DIM * sizeof(float));
    memcpy(buffer_ptr_decoder, joiner_input1, DECODER_DIM * sizeof(float));

    int output_cnt = net.get_output_cnt();
    float **output_data = new float*[output_cnt]();


    status = net.network_input_output_set();
    if (status != 0) {
        printf("Failed to set input/output, status=%d, network_idx=%d\n", status, network_idx);
        goto out;
    }

    status = net.network_run();
    if (status != 0) {
        printf("Fail to run network, status=%d, network_idx=%d\n", status, network_idx);
        goto out;
    }

    net.get_output(output_data);

    if (output_cnt > 0 && output_data[0] != nullptr) {
        size_t data_len = net.m_output_data_len[0];
        if (data_len > (size_t)JOINER_OUTPUT_SIZE) data_len = JOINER_OUTPUT_SIZE;
        memcpy(joiner_output, output_data[0], data_len * sizeof(float));
    }

out:
    for (int i = 0; i < output_cnt; i++) {
        delete[] output_data[i];
        output_data[i] = nullptr;
    }
    delete[] output_data;
    output_data = nullptr;

    return status;
}


struct KwsHyp {
    int ys[2];
    int node;
    float log_prob;
    int trailing_blanks;
    std::vector<float> tok_probs;
};

static void hyp_to_dec_in(const KwsHyp &h, float *dec_in) {
    dec_in[0] = (float)h.ys[0];
    dec_in[1] = (float)h.ys[1];
}

static int read_stdin_s16le(float *out, int nmax)
{
    int16_t tmp[1600];
    int n = nmax < 1600 ? nmax : 1600;
    size_t got = fread(tmp, sizeof(int16_t), (size_t)n, stdin);
    if (got == 0) return 0;
    for (size_t i = 0; i < got; ++i)
        out[i] = (float)tmp[i] / 32768.f;
    return (int)got;
}

static int run_kws(Encoder *encoder_ptr, Decoder *decoder_ptr, Joiner *joiner_ptr,
                   audio_buffer_t *audio, VocabEntry *vocab, const char *keywords_path,
                   std::vector<std::string> *hits, float *audio_length,
                   bool stream_stdin, bool exit_on_hit)
{
    hits->clear();
    std::unordered_map<std::string, int> token2id;
    for (int i = 0; i < VOCAB_NUM; ++i) {
        if (vocab[i].token)
            token2id[vocab[i].token] = vocab[i].index;
    }

    std::vector<KwsKeyword> kws;
    if (load_keywords(keywords_path, token2id, &kws, 1.5f, 0.25f) != 0) {
        printf("load_keywords failed\n");
        return -1;
    }
    KwsGraph graph;
    graph.reset();
    for (auto &k : kws) graph.add_keyword(k);

    int status = 0;
    int num_frames = 0;
    int frame_offset = 0;
    int num_processed_frames = 0;
    int offset = N_OFFSET;
    int segment = N_SEGMENT;
    float tail_pad_length = 0.0f;
    const int max_paths = 8;
    const int num_trailing_need = -1;  /* emit on completing last phoneme */
    int debug_frames = 0;
    int debug = getenv("DEBUG_KWS") && getenv("DEBUG_KWS")[0] == '1';

    knf::FbankOptions fbank_opts;
    fbank_opts.frame_opts.samp_freq = 16000;
    fbank_opts.mel_opts.num_bins = 80;
    fbank_opts.mel_opts.high_freq = -400;
    fbank_opts.frame_opts.dither = 0;
    fbank_opts.frame_opts.snip_edges = false;
    std::unique_ptr<knf::OnlineFbank> fbank(new knf::OnlineFbank(fbank_opts));
    if (!stream_stdin && audio && audio->data) {
        fbank->AcceptWaveform(SAMPLE_RATE, audio->data, audio->num_frames);
        num_frames = fbank->NumFramesReady();
    }

    float *encoder_input = (float *)calloc(ENCODER_INPUT_SIZE, sizeof(float));
    float *encoder_output = (float *)calloc(ENCODER_OUTPUT_SIZE, sizeof(float));
    float *dec_in = (float *)calloc(CONTEXT_SIZE, sizeof(float));
    float *decoder_output = (float *)calloc(DECODER_DIM, sizeof(float));
    float *joiner_output = (float *)calloc(JOINER_OUTPUT_SIZE, sizeof(float));
    float *logits = (float *)calloc(JOINER_OUTPUT_SIZE, sizeof(float));

    float **encoder_input_data = nullptr;
    status = encoder_ptr->init_encoder_input(&encoder_input_data);
    if (status != 0) {
        printf("Encoder init failed! status=%d\n", status);
        goto out;
    }

    {
        std::vector<KwsHyp> beam;
        KwsHyp h0;
        h0.ys[0] = 0;
        h0.ys[1] = 0;
        h0.node = 0;
        h0.log_prob = 0.f;
        h0.trailing_blanks = 0;
        beam.push_back(h0);

        bool flushed = false;
        bool restart_stream = false;
        while (true) {
            if ((num_frames - num_processed_frames) < segment) {
                if (stream_stdin && !flushed) {
                    float pcm[1600];
                    int n = read_stdin_s16le(pcm, 1600);
                    if (n > 0) {
                        fbank->AcceptWaveform(SAMPLE_RATE, pcm, n);
                        num_frames = fbank->NumFramesReady();
                        continue;
                    }
                    /* stdin EOF */
                }
                if (!flushed) {
                    tail_pad_length = (segment - (num_frames - num_processed_frames)) / 100.0f;
                    if (tail_pad_length < 0.f) tail_pad_length = 0.3f;
                    std::vector<float> tail_paddings(int(tail_pad_length * SAMPLE_RATE));
                    fbank->AcceptWaveform(SAMPLE_RATE, tail_paddings.data(), tail_paddings.size());
                    fbank->InputFinished();
                    flushed = true;
                    num_frames = fbank->NumFramesReady();
                }
                if ((num_frames - num_processed_frames) < segment)
                    break;
            }

            status = get_kbank_frames(fbank.get(), num_processed_frames, segment, encoder_input);
            if (status < 0) {
                printf("get_kbank_frames return: %d\n", status);
                break;
            }

            int last = encoder_ptr->plens_index;
            if (last < 0) last = encoder_ptr->net.get_input_cnt() - 1;
            /* Keep streaming caches by default. RESET_CACHE=1 zeros left-context
             * each chunk (debug: if 小瑞 improves, cache feedback is wrong).
             * PLENS_FEEDBACK=1 keeps the model's new_processed_lens (typically
             * +8) instead of overwriting with feature-frame index (0,16,32). */
            static int reset_cache = -1;
            static int plens_feedback = -1;
            if (reset_cache < 0)
                reset_cache = getenv("RESET_CACHE") && getenv("RESET_CACHE")[0] == '1';
            if (plens_feedback < 0)
                plens_feedback = getenv("PLENS_FEEDBACK") && getenv("PLENS_FEEDBACK")[0] == '1';
            if (reset_cache) {
                for (int i = 1; i < last; ++i)
                    memset(encoder_input_data[i], 0,
                           encoder_ptr->cache_bytes[i] ? encoder_ptr->cache_bytes[i]
                           : encoder_ptr->net.m_input_data_len[i] * sizeof(float));
                encoder_ptr->write_processed_lens(encoder_input_data, 0);
            } else if (!plens_feedback) {
                encoder_ptr->write_processed_lens(encoder_input_data,
                                                  num_processed_frames);
            }

            status = encoder_ptr->run_encoder_model(encoder_input, encoder_output, &encoder_input_data);
            if (status < 0) {
                printf("encoder fail\n");
                goto out;
            }

            {
                double s = 0, s2 = 0;
                int n = ENCODER_OUTPUT_SIZE;
                float mn = encoder_output[0], mx = encoder_output[0];
                for (int i = 0; i < n; ++i) {
                    float v = encoder_output[i];
                    s += v; s2 += (double)v * v;
                    if (v < mn) mn = v;
                    if (v > mx) mx = v;
                }
                if (debug)
                    printf("enc chunk@%d x0=%.3f mean=%.4f std=%.4f min=%.3f max=%.3f first8=%.3f %.3f %.3f %.3f\n",
                           num_processed_frames, encoder_input[0],
                           s / n, sqrt(s2 / n - (s / n) * (s / n)), mn, mx,
                           encoder_output[0], encoder_output[1], encoder_output[2], encoder_output[3]);
            }

            for (int t = 0; t < ENCODER_OUTPUT_T; ++t) {
                float *enc_t = encoder_output + t * DECODER_DIM;
                struct Cand {
                    int hyp_i;
                    int tok;
                    float score;
                    int new_node;
                    float boost;
                    float lpt;
                };
                std::vector<Cand> cands;
                std::vector<std::vector<float>> dec_outs(beam.size(), std::vector<float>(DECODER_DIM));

                for (size_t hi = 0; hi < beam.size(); ++hi) {
                    hyp_to_dec_in(beam[hi], dec_in);
                    status = decoder_ptr->run_decoder_model(dec_in, dec_outs[hi].data());
                    if (status < 0) goto out;
                    status = joiner_ptr->run_joiner_model(enc_t, dec_outs[hi].data(), joiner_output);
                    if (status < 0) goto out;
                    memcpy(logits, joiner_output, JOINER_OUTPUT_SIZE * sizeof(float));
                    log_softmax(logits, JOINER_OUTPUT_SIZE);

                    if (debug && hi == 0) {
                        int b = top1(logits, JOINER_OUTPUT_SIZE);
                        if (b != BLANK_ID || debug_frames < 2) {
                            printf("t=%d top='%s' p=%.2f\n",
                                   frame_offset + t,
                                   (b >= 0 && b < VOCAB_NUM && vocab[b].token) ? vocab[b].token : "?",
                                   expf(logits[b]));
                        }
                        ++debug_frames;
                    }

                    /* always keep blank + trie continuation + acoustic top-3 */
                    bool seen[JOINER_OUTPUT_SIZE];
                    memset(seen, 0, sizeof(seen));
                    auto add_tok = [&](int tok) {
                        if (tok < 0 || tok >= JOINER_OUTPUT_SIZE || seen[tok] || tok == UNK_ID)
                            return;
                        seen[tok] = true;
                        float boost = 0.f;
                        int nn = beam[hi].node;
                        if (tok != BLANK_ID)
                            nn = graph.step(beam[hi].node, tok, &boost);
                        float sc = beam[hi].log_prob + logits[tok] + (tok == BLANK_ID ? 0.f : boost);
                        cands.push_back({(int)hi, tok, sc, nn, boost, logits[tok]});
                    };
                    add_tok(BLANK_ID);
                    float best_next_lp = -1e30f;
                    int best_next = -1;
                    for (auto &kv : graph.nodes[beam[hi].node].next) {
                        add_tok(kv.first);
                        if (logits[kv.first] > best_next_lp) {
                            best_next_lp = logits[kv.first];
                            best_next = kv.first;
                        }
                    }
                    /* Extra continuation boost: NPU later frames under-score
                     * r/uì vs CPU. Only if already deep on a path and the next
                     * phoneme is not dead (lp > -9). */
                    if (best_next >= 0 && graph.nodes[beam[hi].node].level >= 3 &&
                        best_next_lp > -9.f) {
                        float boost = 0.f;
                        int nn = graph.step(beam[hi].node, best_next, &boost);
                        float sc = beam[hi].log_prob + best_next_lp + boost + 5.0f;
                        cands.push_back({(int)hi, best_next, sc, nn, boost, best_next_lp});
                    }
                    /* acoustic top-3 */
                    int topid[3] = {0, 0, 0};
                    float topv[3] = {-1e30f, -1e30f, -1e30f};
                    for (int tok = 0; tok < JOINER_OUTPUT_SIZE; ++tok) {
                        float v = logits[tok];
                        if (v > topv[0]) { topv[2]=topv[1]; topid[2]=topid[1]; topv[1]=topv[0]; topid[1]=topid[0]; topv[0]=v; topid[0]=tok; }
                        else if (v > topv[1]) { topv[2]=topv[1]; topid[2]=topid[1]; topv[1]=v; topid[1]=tok; }
                        else if (v > topv[2]) { topv[2]=v; topid[2]=tok; }
                    }
                    for (int k = 0; k < 3; ++k) add_tok(topid[k]);
                }

                /* Per-hyp keep: never drop a keyword path just because blank
                 * from another hyp has higher global score. */
                std::vector<KwsHyp> next;
                bool emitted = false;
                auto apply_cand = [&](const Cand &c) {
                    KwsHyp nh = beam[c.hyp_i];
                    nh.log_prob = c.score;
                    if (c.tok != BLANK_ID && c.tok != UNK_ID) {
                        nh.ys[0] = nh.ys[1];
                        nh.ys[1] = c.tok;
                        nh.node = c.new_node;
                        nh.trailing_blanks = 0;
                        nh.tok_probs.push_back(expf(c.lpt));
                    } else {
                        nh.trailing_blanks += 1;
                    }
                    if (debug && nh.node > 0 && graph.nodes[nh.node].level > 0) {
                        printf("prog t=%d level=%d end=%d tok=%d\n",
                               frame_offset + t, graph.nodes[nh.node].level,
                               graph.nodes[nh.node].is_end ? 1 : 0,
                               graph.nodes[nh.node].token);
                    }
                    if (!emitted && nh.node > 0 && graph.nodes[nh.node].is_end &&
                        nh.trailing_blanks > num_trailing_need) {
                        const TrieNode &nd = graph.nodes[nh.node];
                        float meanp = 0.f;
                        int lv = nd.level;
                        int nuse = (int)nh.tok_probs.size();
                        if (nuse > lv) nuse = lv;
                        if (nuse > 0) {
                            for (int k = 0; k < nuse; ++k)
                                meanp += nh.tok_probs[nh.tok_probs.size() - nuse + k];
                            meanp /= nuse;
                        }
                        int n_strong = 0;
                        for (float p : nh.tok_probs)
                            if (p > 0.12f) n_strong++;
                        if (debug)
                            printf("endchk t=%d '%s' meanp=%.3f strong=%d nuse=%d/%d trail=%d\n",
                                   frame_offset + t, nd.phrase.c_str(), meanp, n_strong, nuse, lv,
                                   nh.trailing_blanks);
                        if (n_strong >= 2 && nuse >= 4) {
                            printf("HIT '%s' score=%.3f frame=%d\n", nd.phrase.c_str(), meanp,
                                   frame_offset + t);
                            fflush(stdout);
                            hits->push_back(nd.phrase);
                            emitted = true;
                            nh = h0;
                        }
                    }
                    next.push_back(nh);
                };

                for (size_t hi = 0; hi < beam.size(); ++hi) {
                    const Cand *best_blank = nullptr, *best_next = nullptr, *best_any = nullptr;
                    for (auto &c : cands) {
                        if (c.hyp_i != (int)hi) continue;
                        if (!best_any || c.score > best_any->score) best_any = &c;
                        if (c.tok == BLANK_ID) {
                            if (!best_blank || c.score > best_blank->score) best_blank = &c;
                        } else if (graph.nodes[beam[hi].node].next.count(c.tok)) {
                            if (!best_next || c.score > best_next->score) best_next = &c;
                        }
                    }
                    bool take_next = false;
                    if (best_next && best_blank) {
                        if (beam[hi].node > 0)
                            take_next = best_next->score >= best_blank->score;
                        else
                            take_next = (expf(best_next->lpt) > 0.08f) &&
                                        (best_next->score >= best_blank->score);
                    }

                    if (take_next)
                        apply_cand(*best_next);
                    else if (best_blank)
                        apply_cand(*best_blank);
                    else if (best_any)
                        apply_cand(*best_any);
                }
                if (next.empty()) next.push_back(h0);
                beam.swap(next);
                if (emitted) {
                    beam.clear();
                    beam.push_back(h0);
                    if (stream_stdin) {
                        if (exit_on_hit)
                            goto out;
                        int nio = encoder_ptr->net.get_input_cnt();
                        int plast = encoder_ptr->plens_index;
                        if (plast < 0) plast = nio - 1;
                        for (int i = 1; i < plast; ++i)
                            memset(encoder_input_data[i], 0,
                                   encoder_ptr->cache_bytes[i] ? encoder_ptr->cache_bytes[i]
                                   : encoder_ptr->net.m_input_data_len[i] * sizeof(float));
                        encoder_ptr->write_processed_lens(encoder_input_data, 0);
                        fbank.reset(new knf::OnlineFbank(fbank_opts));
                        num_frames = 0;
                        num_processed_frames = 0;
                        frame_offset = 0;
                        flushed = false;
                        restart_stream = true;
                        break;
                    }
                }
            }
            if (restart_stream) {
                restart_stream = false;
                continue;
            }
            frame_offset += ENCODER_OUTPUT_T;
            num_processed_frames += offset;
        }
    }

    if (audio_length) {
        if (audio && audio->sample_rate > 0)
            *audio_length = (float)audio->num_frames / audio->sample_rate + tail_pad_length;
        else
            *audio_length = 0.f;
    }

out:
    free(encoder_input);
    free(encoder_output);
    free(dec_in);
    free(decoder_output);
    free(joiner_output);
    free(logits);

    if (encoder_input_data) {
        int encoder_input_cnt = encoder_ptr->net.get_input_cnt();
        for (int i = 0; i < encoder_input_cnt; i++)
            delete[] encoder_input_data[i];
        delete[] encoder_input_data;
    }
    return status;
}

int inference_kws_model(Encoder *encoder_ptr, Decoder *decoder_ptr, Joiner *joiner_ptr, audio_buffer_t audio,
                        VocabEntry *vocab, const char *keywords_path,
                        std::vector<std::string> *hits, float &audio_length)
{
    return run_kws(encoder_ptr, decoder_ptr, joiner_ptr, &audio, vocab, keywords_path,
                   hits, &audio_length, false, false);
}

int inference_kws_stream(Encoder *encoder_ptr, Decoder *decoder_ptr, Joiner *joiner_ptr,
                         VocabEntry *vocab, const char *keywords_path,
                         std::vector<std::string> *hits, bool exit_on_hit)
{
    float dummy = 0.f;
    return run_kws(encoder_ptr, decoder_ptr, joiner_ptr, nullptr, vocab, keywords_path,
                   hits, &dummy, true, exit_on_hit);
}


