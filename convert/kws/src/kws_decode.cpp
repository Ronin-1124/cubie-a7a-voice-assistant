#include "kws_decode.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <sstream>

void KwsGraph::reset() {
    nodes.clear();
    TrieNode root;
    root.token = -1;
    nodes.push_back(root);
}

void KwsGraph::add_keyword(const KwsKeyword &kw) {
    if (nodes.empty()) reset();
    int cur = 0;
    for (size_t i = 0; i < kw.tokens.size(); ++i) {
        int t = kw.tokens[i];
        auto it = nodes[cur].next.find(t);
        if (it == nodes[cur].next.end()) {
            TrieNode n;
            n.token = t;
            n.level = nodes[cur].level + 1;
            n.boost = kw.boost;
            nodes.push_back(n);
            int ni = (int)nodes.size() - 1;
            nodes[cur].next[t] = ni;
            cur = ni;
        } else {
            cur = it->second;
        }
    }
    nodes[cur].is_end = true;
    nodes[cur].phrase = kw.phrase;
    nodes[cur].threshold = kw.threshold;
    nodes[cur].boost = kw.boost;
}

int KwsGraph::step(int node, int token, float *boost_out) const {
    if (node < 0 || node >= (int)nodes.size()) node = 0;
    auto it = nodes[node].next.find(token);
    if (it != nodes[node].next.end()) {
        if (boost_out) *boost_out = nodes[it->second].boost;
        return it->second;
    }
    // fail to root, try start of another keyword
    if (node != 0) {
        auto it2 = nodes[0].next.find(token);
        if (it2 != nodes[0].next.end()) {
            if (boost_out) *boost_out = nodes[it2->second].boost;
            return it2->second;
        }
    }
    if (boost_out) *boost_out = 0.f;
    return 0;
}

int load_keywords(const char *path, const std::unordered_map<std::string, int> &token2id,
                  std::vector<KwsKeyword> *out, float default_boost, float default_thr) {
    out->clear();
    std::ifstream ifs(path);
    if (!ifs) {
        printf("open keywords failed: %s\n", path);
        return -1;
    }
    std::string line;
    int lineno = 0;
    while (std::getline(ifs, line)) {
        ++lineno;
        if (line.empty() || line[0] == '#') continue;
        float boost = default_boost, thr = default_thr;
        std::string phrase;
        std::string rest = line;
        auto at = rest.find('@');
        if (at != std::string::npos) {
            phrase = rest.substr(at + 1);
            while (!phrase.empty() && (phrase.back() == '\r' || phrase.back() == ' '))
                phrase.pop_back();
            rest = rest.substr(0, at);
        }
        auto hp = rest.find('#');
        if (hp != std::string::npos) {
            thr = (float)atof(rest.c_str() + hp + 1);
            rest = rest.substr(0, hp);
        }
        auto cl = rest.find(':');
        if (cl != std::string::npos) {
            boost = (float)atof(rest.c_str() + cl + 1);
            rest = rest.substr(0, cl);
        }
        std::istringstream iss(rest);
        std::string tok;
        KwsKeyword kw;
        kw.boost = boost;
        kw.threshold = thr;
        while (iss >> tok) {
            auto it = token2id.find(tok);
            if (it == token2id.end()) {
                printf("keywords L%d unknown token '%s'\n", lineno, tok.c_str());
                kw.tokens.clear();
                break;
            }
            kw.tokens.push_back(it->second);
        }
        if (kw.tokens.empty()) continue;
        if (phrase.empty()) {
            for (int id : kw.tokens) phrase += std::to_string(id) + " ";
        }
        kw.phrase = phrase;
        printf("  kw '%s' tokens=%zu boost=%.2f thr=%.2f\n",
               kw.phrase.c_str(), kw.tokens.size(), kw.boost, kw.threshold);
        out->push_back(kw);
    }
    return out->empty() ? -2 : 0;
}

void log_softmax(float *x, int n) {
    float m = x[0];
    for (int i = 1; i < n; ++i)
        if (x[i] > m) m = x[i];
    float s = 0.f;
    for (int i = 0; i < n; ++i) {
        x[i] = expf(x[i] - m);
        s += x[i];
    }
    float inv = 1.f / (s > 0.f ? s : 1.f);
    for (int i = 0; i < n; ++i) {
        float p = x[i] * inv;
        x[i] = (p > 1e-12f) ? logf(p) : -30.f;
    }
}

int top1(const float *x, int n) {
    int b = 0;
    for (int i = 1; i < n; ++i)
        if (x[i] > x[b]) b = i;
    return b;
}
