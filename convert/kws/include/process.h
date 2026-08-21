#ifndef _AW_ZIPFORMER_DEMO_PROCESS_H_
#define _AW_ZIPFORMER_DEMO_PROCESS_H_

// #define TIMING_DISABLED // if you don't need to print the time used, uncomment this line of code
#include "online-feature.h"


#define VOCAB_NUM 263
#define SAMPLE_RATE 16000
#define N_MELS 80
#define N_SEGMENT 29
#define ENCODER_OUTPUT_T 4
#define DECODER_DIM 320
#define ENCODER_INPUT_SIZE N_MELS *N_SEGMENT
#define ENCODER_OUTPUT_SIZE ENCODER_OUTPUT_T *DECODER_DIM
#define JOINER_OUTPUT_SIZE 263
#define N_OFFSET 16
#define CONTEXT_SIZE 2
#define MAX_ENCODER_IO 64

#define VOCAB_PATH "./model/tokens.txt"

typedef struct
{
    int index;
    char *token;
} VocabEntry;

int get_kbank_frames(knf::OnlineFbank *fbank, int frame_index, int segment, float *frames);
void convert_nchw_to_nhwc(float *src, float *dst, int N, int channels, int height, int width);
int argmax(float *array);
void replace_substr(std::string &str, const std::string &from, const std::string &to);
int read_vocab(const char *fileName, VocabEntry *vocab);

#endif //_AW_ZIPFORMER_DEMO_PROCESS_H_
