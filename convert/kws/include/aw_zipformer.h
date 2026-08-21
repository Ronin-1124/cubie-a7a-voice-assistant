#ifndef _AW_DEMO_ZIPFORMER_H_
#define _AW_DEMO_ZIPFORMER_H_


#include <iostream>
#include <vector>
#include <string>

#include "process.h"
#include "npulib.h"
#include "audio_utils.h"

#define BLANK_ID 0
#define UNK_ID 2


class Encoder
{
public:
    Encoder(const char* model_path);
    ~Encoder();

    int init_encoder_input(float ***input_data_ptr);
    int run_encoder_model(float *encoder_input, float *encoder_output, float ***encoder_init_data);
    /* Write processed_lens in the VIP native dtype (INT32/INT64/FP32). */
    void write_processed_lens(float **input_data, int value);

public:
    NetworkItem net;
    int network_idx;
    unsigned int cache_bytes[MAX_ENCODER_IO];
    int plens_index;
};


class Decoder
{
public:
    Decoder(const char* model_path);
    ~Decoder();

    int run_decoder_model(float *decoder_input, float *decoder_output);

public:
    NetworkItem net;
    int network_idx;
};


class Joiner
{
public:
    Joiner(const char* model_path);
    ~Joiner();

    int run_joiner_model(float* joiner_input0, float* joiner_input1, float *joiner_output);

public:
    NetworkItem net;
    int network_idx;
};


int inference_kws_model(Encoder *encoder_ptr, Decoder *decoder_ptr, Joiner *joiner_ptr, audio_buffer_t audio,
                        VocabEntry *vocab, const char *keywords_path,
                        std::vector<std::string> *hits, float &audio_length);

/* stream_stdin: read raw S16LE 16 kHz mono from stdin. exit_on_hit: stop after first HIT. */
int inference_kws_stream(Encoder *encoder_ptr, Decoder *decoder_ptr, Joiner *joiner_ptr,
                         VocabEntry *vocab, const char *keywords_path,
                         std::vector<std::string> *hits, bool exit_on_hit);

#endif //_AW_DEMO_ZIPFORMER_H_