#ifndef KWS_DECODE_H
#define KWS_DECODE_H

#include <string>
#include <vector>
#include <unordered_map>

struct KwsKeyword {
    std::vector<int> tokens;
    std::string phrase;
    float boost;
    float threshold;
};

struct KwsHit {
    std::string phrase;
    float score;
    int frame;
};

struct TrieNode {
    int token = -1;
    int level = 0;
    bool is_end = false;
    float boost = 1.0f;
    float threshold = 0.25f;
    std::string phrase;
    std::unordered_map<int, int> next;  // token -> node index
};

struct KwsGraph {
    std::vector<TrieNode> nodes;  // 0 = root
    void reset();
    void add_keyword(const KwsKeyword &kw);
    int step(int node, int token, float *boost_out) const;
};

int load_keywords(const char *path, const std::unordered_map<std::string, int> &token2id,
                  std::vector<KwsKeyword> *out, float default_boost, float default_thr);

void log_softmax(float *x, int n);

int top1(const float *x, int n);

#endif
