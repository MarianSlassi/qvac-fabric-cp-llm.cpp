// End-to-end checks for the hybrid Gated DeltaNet + NSA model.
//
// Set NSA_TEST_MODEL to the GGUF file. The test is skipped when the variable is
// not set, so normal CI runs do not need the 1.4 GB checkpoint.
//
// It checks that prompt processing (all tokens in one batch) and token-by-token
// generation produce the same logits. This catches most cache/window/block bugs.

#include "llama.h"
#include "llama-ext.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

static std::vector<llama_token> tokenize(const llama_vocab * vocab, const std::string & text) {
    int32_t n = llama_tokenize(vocab, text.c_str(), (int32_t) text.size(), nullptr, 0, true, true);
    if (n < 0) {
        n = -n;
    }
    std::vector<llama_token> tokens(n);
    n = llama_tokenize(vocab, text.c_str(), (int32_t) text.size(), tokens.data(), (int32_t) tokens.size(), true, true);
    if (n < 0) {
        fprintf(stderr, "tokenize failed\n");
        exit(1);
    }
    tokens.resize(n);
    return tokens;
}

static bool dump_logits(const char * path, int32_t n_tokens, int32_t n_vocab, const std::vector<float> & logits) {
    FILE * f = fopen(path, "wb");
    if (!f) {
        return false;
    }
    fwrite(&n_tokens, sizeof(n_tokens), 1, f);
    fwrite(&n_vocab,  sizeof(n_vocab),  1, f);
    fwrite(logits.data(), sizeof(float), logits.size(), f);
    fclose(f);
    return true;
}

int main(int argc, char ** argv) {
    const char * model_path = getenv("NSA_TEST_MODEL");
    if (!model_path || model_path[0] == '\0') {
        fprintf(stderr, "NSA_TEST_MODEL is not set, skipping\n");
        return 77;
    }

    const char * prompt = getenv("NSA_TEST_PROMPT");
    if (!prompt) {
        prompt = "The capital of France is Paris. It is known for the Eiffel Tower and the Louvre museum.";
    }

    const char * dump_path = nullptr;
    if (argc > 1) {
        dump_path = argv[1];
    }

    llama_backend_init();

    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = 0;
    if (const char * env = getenv("NSA_TEST_GPU_LAYERS")) {
        mparams.n_gpu_layers = atoi(env);
    }

    llama_model * model = llama_model_load_from_file(model_path, mparams);
    if (!model) {
        fprintf(stderr, "failed to load model\n");
        return 1;
    }

    const llama_vocab * vocab = llama_model_get_vocab(model);
    const int32_t n_vocab = llama_vocab_n_tokens(vocab);
    std::vector<llama_token> tokens;
    if (const char * env = getenv("NSA_TEST_FAKE_TOKENS")) {
        const int n = atoi(env);
        tokens.reserve(n);
        tokens.push_back(1);
        for (int i = 1; i < n; ++i) {
            tokens.push_back(100 + (i*37) % 20000);
        }
    } else {
        tokens = tokenize(vocab, prompt);
    }
    const int32_t n_tokens = (int32_t) tokens.size();

    printf("prompt tokens: %d\n", n_tokens);
    for (int i = 0; i < n_tokens; ++i) {
        printf("%d ", tokens[i]);
    }
    printf("\n");

    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx           = 512;
    cparams.n_batch         = 512;
    cparams.n_ubatch        = 512;
    cparams.n_seq_max       = 1;
    cparams.n_threads       = 4;
    cparams.n_threads_batch = 4;
    cparams.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
    if (getenv("NSA_TEST_KV_F32")) {
        cparams.type_k = GGML_TYPE_F32;
        cparams.type_v = GGML_TYPE_F32;
    }

    const bool dump_layers = getenv("NSA_DUMP_LAYERS") != nullptr;

    llama_context * ctx = llama_init_from_model(model, cparams);
    if (!ctx) {
        fprintf(stderr, "failed to create context\n");
        return 1;
    }

    int32_t chunk = 0;
    if (const char * env = getenv("NSA_TEST_PREFILL_CHUNK")) {
        chunk = atoi(env);
    }

    std::vector<float> prefill((size_t) n_tokens * n_vocab);
    if (chunk <= 0 || chunk >= n_tokens) {
        if (dump_layers) {
        for (uint32_t il = 0; il < llama_model_n_layer(model); ++il) {
            llama_set_embeddings_layer_inp(ctx, il, true);
        }
    }

    llama_batch batch = llama_batch_init(n_tokens, 0, 1);
        batch.n_tokens = n_tokens;
        for (int32_t i = 0; i < n_tokens; ++i) {
            batch.token[i]    = tokens[i];
            batch.pos[i]      = i;
            batch.n_seq_id[i] = 1;
            batch.seq_id[i][0] = 0;
            batch.logits[i]   = 1;
        }

        if (llama_decode(ctx, batch) != 0) {
            fprintf(stderr, "prefill decode failed\n");
            return 1;
        }

        for (int32_t i = 0; i < n_tokens; ++i) {
            const float * logits = llama_get_logits_ith(ctx, i);
            memcpy(prefill.data() + (size_t) i * n_vocab, logits, sizeof(float) * n_vocab);
        }
        llama_batch_free(batch);
    } else {
        for (int32_t i0 = 0; i0 < n_tokens; i0 += chunk) {
            const int32_t n = std::min(chunk, n_tokens - i0);
            llama_batch batch = llama_batch_init(n, 0, 1);
            batch.n_tokens = n;
            for (int32_t i = 0; i < n; ++i) {
                batch.token[i]    = tokens[i0 + i];
                batch.pos[i]      = i0 + i;
                batch.n_seq_id[i] = 1;
                batch.seq_id[i][0] = 0;
                batch.logits[i]   = 1;
            }
            if (llama_decode(ctx, batch) != 0) {
                fprintf(stderr, "prefill decode failed at %d\n", i0);
                return 1;
            }
            for (int32_t i = 0; i < n; ++i) {
                const float * logits = llama_get_logits_ith(ctx, i);
                memcpy(prefill.data() + (size_t) (i0 + i) * n_vocab, logits, sizeof(float) * n_vocab);
            }
            llama_batch_free(batch);
        }
    }

    if (dump_layers) {
        const int n_embd = llama_model_n_embd(model);
        for (uint32_t il = 0; il < llama_model_n_layer(model); ++il) {
            float * p = llama_get_embeddings_layer_inp(ctx, il);
            char name[256];
            snprintf(name, sizeof(name), "%s/cpp_layer_%02u.f32", getenv("NSA_DUMP_LAYERS"), il);
            FILE * f = fopen(name, "wb");
            if (f) {
                fwrite(p, sizeof(float), (size_t) n_embd * n_tokens, f);
                fclose(f);
            }
        }
    }

    if (dump_path) {
        if (!dump_logits(dump_path, n_tokens, n_vocab, prefill)) {
            fprintf(stderr, "failed to write %s\n", dump_path);
            return 1;
        }
        printf("wrote %s\n", dump_path);
    }

    // token-by-token pass
    llama_memory_clear(llama_get_memory(ctx), true);

    std::vector<float> decode_logits((size_t) n_tokens * n_vocab);

    double max_diff = 0.0;
    double sum_diff = 0.0;
    int    n_diff   = 0;
    int    top1_bad = 0;

    for (int32_t i = 0; i < n_tokens; ++i) {
        llama_batch one = llama_batch_init(1, 0, 1);
        one.n_tokens = 1;
        one.token[0] = tokens[i];
        one.pos[0] = i;
        one.n_seq_id[0] = 1;
        one.seq_id[0][0] = 0;
        one.logits[0] = 1;

        if (llama_decode(ctx, one) != 0) {
            fprintf(stderr, "decode failed at token %d\n", i);
            return 1;
        }
        llama_batch_free(one);

        const float * logits = llama_get_logits_ith(ctx, 0);
        memcpy(decode_logits.data() + (size_t) i * n_vocab, logits, sizeof(float) * n_vocab);
        const float * ref    = prefill.data() + (size_t) i * n_vocab;

        int top_ref = 0;
        int top_cur = 0;
        double max_diff_i = 0.0;
        for (int32_t v = 0; v < n_vocab; ++v) {
            const double d = std::fabs((double) logits[v] - (double) ref[v]);
            max_diff = std::max(max_diff, d);
            max_diff_i = std::max(max_diff_i, d);
            sum_diff += d;
            n_diff++;
            if (ref[v] > ref[top_ref]) {
                top_ref = v;
            }
            if (logits[v] > logits[top_cur]) {
                top_cur = v;
            }
        }
        if (max_diff_i > 0.25) {
            printf("token %d: max diff %.4f\n", i, max_diff_i);
        }
        if (top_ref != top_cur) {
            top1_bad++;
            if (top1_bad < 8) {
                printf("token %d: prefill top %d, decode top %d\n", i, top_ref, top_cur);
            }
        }
    }

    const double mean_diff = n_diff ? sum_diff / n_diff : 0.0;
    printf("prefill vs decode: max abs diff %.6g, mean abs diff %.6g, top1 mismatches %d/%d\n",
           max_diff, mean_diff, top1_bad, n_tokens);

    if (const char * path = getenv("NSA_DUMP_DECODE")) {
        dump_logits(path, n_tokens, n_vocab, decode_logits);
        printf("wrote %s\n", path);
    }

    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();

    // CPU is the reference. GPU paths may differ by one token at long lengths
    // because of f32 accumulation in the backend kernels.
    const int top1_tol = mparams.n_gpu_layers > 0 ? 1 : 0;
    if (max_diff > 5e-2 || top1_bad > top1_tol) {
        fprintf(stderr, "FAILED\n");
        return 1;
    }
    printf("PASSED\n");
    return 0;
}
