#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <iostream>
#include <vector>
#include <string>
#include <iomanip>

#include "process.h"
#include "audio_utils.h"
#include "aw_zipformer.h"
#include <chrono>


const char *usage =
    "kws_npu_demo -nb0 encoder -nb1 decoder -nb2 joiner -i wav [-k keywords.txt]\n"
    "  --stdin          raw S16LE 16kHz mono on stdin (assistant streaming)\n"
    "  --exit-on-hit    with --stdin, exit after the first HIT\n";


int main(int argc, char **argv)
{
    int i = 0, status = 0;
    char *encoder_path = nullptr;
    char *decoder_path = nullptr;
    char *joiner_path = nullptr;
    char *audio_path = nullptr;
    const char *keywords_path = "./model/keywords_main.txt";
    int stdin_stream = 0;
    int exit_on_hit = 0;

    for (i = 0; i< argc; i++) {
        if (!strcmp(argv[i], "-nb0")) {
            encoder_path = argv[++i];
        }
        else if (!strcmp(argv[i], "-nb1")) {
            decoder_path = argv[++i];
        }
        else if (!strcmp(argv[i], "-nb2")) {
            joiner_path = argv[++i];
        }
        else if (!strcmp(argv[i], "-i")) {
            audio_path = argv[++i];
            if (audio_path && !strcmp(audio_path, "-"))
                stdin_stream = 1;
        }
        else if (!strcmp(argv[i], "-k")) {
            keywords_path = argv[++i];
        }
        else if (!strcmp(argv[i], "--stdin")) {
            stdin_stream = 1;
        }
        else if (!strcmp(argv[i], "--exit-on-hit")) {
            exit_on_hit = 1;
        }
        else if (!strcmp(argv[i], "-h")) {
            printf("%s\n", usage);
            return 0;
        }
    }
    printf("encoder=%s decoder=%s joiner=%s wav=%s kw=%s stdin=%d\n",
           encoder_path, decoder_path, joiner_path,
           audio_path ? audio_path : "-", keywords_path, stdin_stream);
    if (!encoder_path || !decoder_path || !joiner_path) {
        printf("%s\n", usage);
        return 1;
    }
    if (!stdin_stream && !audio_path) {
        printf("need -i wav or --stdin\n");
        return 1;
    }

    float infer_time = 0.0;
    float audio_length = 0.0;
    float rtf = 0.0;
    std::vector<std::string> hits;
    VocabEntry vocab[VOCAB_NUM];
    audio_buffer_t audio;
    memset(vocab, 0, sizeof(vocab));
    memset(&audio, 0, sizeof(audio_buffer_t));

    if (!stdin_stream) {
        status = read_audio(audio_path, &audio);
        if (status != 0) {
            printf("read audio fail! status=%d audio_path=%s\n", status, audio_path);
            return -1;
        }

        if (audio.num_channels == 2) {
            status = convert_channels(&audio);
            if (status != 0) {
                printf("convert channels fail! status=%d audio_path=%s\n", status, audio_path);
                return -1;
            }
        }

        if (audio.sample_rate != SAMPLE_RATE) {
            status = resample_audio(&audio, audio.sample_rate, SAMPLE_RATE);
            if (status != 0) {
                printf("resample audio fail! status=%d audio_path=%s\n", status, audio_path);
                return -1;
            }
        }
    }

    status = read_vocab(VOCAB_PATH, vocab);
    if (status != 0) {
        printf("read vocab fail! status=%d vocab_path=%s\n", status, VOCAB_PATH);
        return -1;
    }


    // NPU init
    NpuUint npu_uint;
    unsigned int version = npu_uint.get_driver_version();
    printf("npu driver version=0x%08x...\n", version);
    //int status = npu_uint.npu_init(malloc_mbyte*1024*1024);    // 85x
    status = npu_uint.npu_init();
    if (status != 0) {
        return -1;
    }

    Encoder *encoder_ptr = new Encoder((const char*)encoder_path);
    Decoder *decoder_ptr = new Decoder((const char*)decoder_path);
    Joiner *joiner_ptr = new Joiner((const char*)joiner_path);

    std::chrono::steady_clock::time_point Tbegin, Tend;
    Tbegin = std::chrono::steady_clock::now();

    if (stdin_stream) {
        setvbuf(stdin, nullptr, _IONBF, 0);
        setvbuf(stdout, nullptr, _IOLBF, 0);
        printf("KWS_READY\n");
        fflush(stdout);
        status = inference_kws_stream(encoder_ptr, decoder_ptr, joiner_ptr, vocab,
                                      keywords_path, &hits, exit_on_hit);
        if (status != 0) {
            printf("inference_kws_stream fail! status=%d\n", status);
            return -1;
        }
    } else {
        status = inference_kws_model(encoder_ptr, decoder_ptr, joiner_ptr, audio, vocab, keywords_path, &hits, audio_length);
        if (status != 0) {
            printf("inference_kws_model fail! status=%d\n", status);
            return -1;
        }

        Tend = std::chrono::steady_clock::now();
        infer_time = std::chrono::duration_cast <std::chrono::milliseconds> (Tend - Tbegin).count() / 1000.0;
        rtf = infer_time / std::max(audio_length, 1e-6f);
        printf("\nRTF: %.3f = %.3fs / %.3fs\n", rtf, infer_time, audio_length);
        if (hits.empty()) {
            printf("HITS: (none)\n");
        } else {
            printf("HITS:");
            for (auto &h : hits) printf(" [%s]", h.c_str());
            printf("\n");
        }
    }

    // free
    if (encoder_ptr != nullptr) {
        delete encoder_ptr;
        encoder_ptr = nullptr;
    }
    if (decoder_ptr != nullptr) {
        delete decoder_ptr;
        decoder_ptr = nullptr;
    }
    if (joiner_ptr != nullptr) {
        delete joiner_ptr;
        joiner_ptr = nullptr;
    }

    return 0;
}
