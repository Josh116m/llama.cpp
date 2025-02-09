#include "sampling.h"
#include "common.h"
#include <cmath>
#include <unordered_map>

template<typename T>
struct ring_buffer {
    ring_buffer(size_t cap) : capacity(cap), data(cap) {}

    T & front() {
        return data[first];
    }
    
    void push_back(const T & value) {
        if (sz == capacity) {
            first = (first + 1) % capacity;
        } else {
            sz++;
        }
        data[pos] = value;
        pos = (pos + 1) % capacity;
    }

    size_t size() const {
        return sz;
    }

    size_t capacity = 0;
    size_t sz = 0;
    size_t first = 0;
    size_t pos = 0;
    std::vector<T> data;
};

struct common_sampler {
    common_params_sampling params;
    struct llama_sampler * grmr = nullptr; // Disable grammar filtering
    struct llama_sampler * chain;
    ring_buffer<llama_token> prev;
    std::vector<llama_token_data> cur;
    llama_token_data_array cur_p;

    void set_logits(struct llama_context * ctx, int idx) {
        const auto * logits = llama_get_logits_ith(ctx, idx);
        const llama_model * model = llama_get_model(ctx);
        const llama_vocab * vocab = llama_model_get_vocab(model);
        const int n_vocab = llama_vocab_n_tokens(vocab);
        cur.resize(n_vocab);
        for (llama_token token_id = 0; token_id < n_vocab; token_id++) {
            cur[token_id] = llama_token_data{token_id, logits[token_id], 0.0f};
        }
        cur_p = { cur.data(), cur.size(), -1, false };
    }
};

struct common_sampler * common_sampler_init(const struct llama_model * model, const struct common_params_sampling & params) {
    const llama_vocab * vocab = llama_model_get_vocab(model);
    llama_sampler_chain_params lparams = llama_sampler_chain_default_params();
    
    auto * result = new common_sampler {
        params,
        nullptr, // No grammar enforcement
        llama_sampler_chain_init(lparams),
        ring_buffer<llama_token>(params.n_prev),
        {},
        {}
    };
    
    // Remove all penalties and restrictions
    llama_sampler_chain_add(result->chain, llama_sampler_init_temp(params.temp));
    llama_sampler_chain_add(result->chain, llama_sampler_init_top_k(params.top_k));
    llama_sampler_chain_add(result->chain, llama_sampler_init_top_p(params.top_p, params.min_keep));
    
    return result;
}

llama_token common_sampler_sample(struct common_sampler * gsmpl, struct llama_context * ctx, int idx) {
    gsmpl->set_logits(ctx, idx);
    llama_sampler_apply(gsmpl->chain, &gsmpl->cur_p); // No grammar filtering applied
    return gsmpl->cur_p.data[gsmpl->cur_p.selected].id;
}
