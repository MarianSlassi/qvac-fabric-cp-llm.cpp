#pragma once

#include "llama-memory.h"

#include <cstdint>
#include <functional>
#include <memory>
#include <vector>

struct ggml_context;
struct ggml_tensor;
struct ggml_backend_buffer;
struct llama_model;

// Per-layer mean K/V of every completed NSA block. One cell is one block.
// The buffer is updated by the graph when a block becomes complete.
class llama_memory_nsa_summary {
public:
    using layer_filter_cb = std::function<bool(uint32_t)>;

    llama_memory_nsa_summary(
            const llama_model & model,
                     uint32_t   n_blocks,
                     uint32_t   n_seqs,
        const layer_filter_cb & filter);

    ~llama_memory_nsa_summary();

    ggml_tensor * get_k(uint32_t il) const;
    ggml_tensor * get_v(uint32_t il) const;

    void clear(bool data);

    size_t size() const;
    std::map<ggml_backend_buffer_type_t, size_t> breakdown() const;

    void state_write(llama_io_write_i & io) const;
    void state_read (llama_io_read_i  & io);

private:
    struct layer {
        ggml_context * ctx = nullptr;
        ggml_backend_buffer * buf = nullptr;
        ggml_tensor * k = nullptr;
        ggml_tensor * v = nullptr;
    };

    std::vector<uint32_t> ids;
    std::vector<layer> layers;
};
