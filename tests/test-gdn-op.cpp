// Focused test for GGML_OP_GATED_DELTA_NET. The model uses the same op for
// every even layer, so a bug here shows up as a prefill/decode mismatch.

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

static constexpr int S = 128;
static constexpr int H = 8;
static constexpr int NS = 1;

static float rnd(std::mt19937 & rng) {
    return std::uniform_real_distribution<float>(-1.0f, 1.0f)(rng);
}

int main() {
    std::mt19937 rng(42);

    const int lengths[] = { 1, 64, 128, 192, 193, 256, 512 };
    int fails = 0;

    for (int n_tokens : lengths) {
        const size_t nq = (size_t) S * H * n_tokens * NS;
        const size_t nst = (size_t) S * S * H * NS;

        std::vector<float> q(nq), k(nq), v(nq);
        std::vector<float> g((size_t) H * n_tokens * NS);
        std::vector<float> beta((size_t) H * n_tokens * NS);
        std::vector<float> state(nst);
        for (float & x : q) x = 0.1f * rnd(rng);
        for (float & x : k) x = 0.1f * rnd(rng);
        for (float & x : v) x = 0.1f * rnd(rng);
        for (float & x : g) x = -0.01f - 0.09f * (rnd(rng) * 0.5f + 0.5f);
        for (float & x : beta) x = 0.01f + 0.09f * (rnd(rng) * 0.5f + 0.5f);
        for (float & x : state) x = 0.01f * rnd(rng);

        struct ggml_init_params ip = { 1024u*1024*1024, nullptr, true };
        struct ggml_context * ctx = ggml_init(ip);
        struct ggml_tensor * tq  = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, S, H, n_tokens, NS);
        struct ggml_tensor * tk  = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, S, H, n_tokens, NS);
        struct ggml_tensor * tv  = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, S, H, n_tokens, NS);
        struct ggml_tensor * tg  = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 1, H, n_tokens, NS);
        struct ggml_tensor * tb  = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 1, H, n_tokens, NS);
        struct ggml_tensor * ts  = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, S, S, H, NS);
        struct ggml_tensor * out = ggml_gated_delta_net(ctx, tq, tk, tv, tg, tb, ts, 1);

        struct ggml_cgraph * gf = ggml_new_graph(ctx);
        ggml_build_forward_expand(gf, out);
        struct ggml_backend * be = ggml_backend_cpu_init();
        ggml_backend_cpu_set_n_threads(be, 6);
        struct ggml_backend_buffer * buf = ggml_backend_alloc_ctx_tensors(ctx, be);
        (void) buf;

        ggml_backend_tensor_set(tq, q.data(),     0, ggml_nbytes(tq));
        ggml_backend_tensor_set(tk, k.data(),     0, ggml_nbytes(tk));
        ggml_backend_tensor_set(tv, v.data(),     0, ggml_nbytes(tv));
        ggml_backend_tensor_set(tg, g.data(),     0, ggml_nbytes(tg));
        ggml_backend_tensor_set(tb, beta.data(),  0, ggml_nbytes(tb));
        ggml_backend_tensor_set(ts, state.data(), 0, ggml_nbytes(ts));

        if (ggml_backend_graph_compute(be, gf) != GGML_STATUS_SUCCESS) {
            printf("FAIL: graph compute n_tokens=%d\n", n_tokens);
            fails++;
            ggml_backend_buffer_free(buf);
            ggml_backend_free(be);
            ggml_free(ctx);
            continue;
        }

        std::vector<float> got((size_t) S * H * (n_tokens + S));
        ggml_backend_tensor_get(out, got.data(), 0, ggml_nbytes(out));

        // naive reference using the same transposed state layout as the kernel
        std::vector<float> ref((size_t) S * H * (n_tokens + S), 0.0f);
        const float scale = 1.0f / sqrtf((float) S);
        for (int ns = 0; ns < NS; ++ns) {
            for (int h = 0; h < H; ++h) {
                std::vector<float> M((size_t) S * S);
                for (int j = 0; j < S; ++j) {
                    for (int i = 0; i < S; ++i) {
                        M[(size_t) j * S + i] = state[(size_t) i + (size_t) S * (j + (size_t) S * (h + (size_t) H * ns))];
                    }
                }
                std::vector<float> delta(S);
                for (int t = 0; t < n_tokens; ++t) {
                    const float * qt = q.data() + (size_t) S * (h + (size_t) H * (t + (size_t) n_tokens * ns));
                    const float * kt = k.data() + (size_t) S * (h + (size_t) H * (t + (size_t) n_tokens * ns));
                    const float * vt = v.data() + (size_t) S * (h + (size_t) H * (t + (size_t) n_tokens * ns));
                    const float gt = g[(size_t) h + (size_t) H * (t + (size_t) n_tokens * ns)];
                    const float bt = beta[(size_t) h + (size_t) H * (t + (size_t) n_tokens * ns)];
                    const float decay = expf(gt);
                    for (int j = 0; j < S; ++j) {
                        for (int i = 0; i < S; ++i) M[(size_t) j * S + i] *= decay;
                    }
                    for (int j = 0; j < S; ++j) {
                        float sum = 0.0f;
                        for (int i = 0; i < S; ++i) sum += M[(size_t) j * S + i] * kt[i];
                        delta[j] = (vt[j] - sum) * bt;
                    }
                    for (int j = 0; j < S; ++j) {
                        for (int i = 0; i < S; ++i) M[(size_t) j * S + i] += kt[i] * delta[j];
                    }
                    for (int j = 0; j < S; ++j) {
                        float sum = 0.0f;
                        for (int i = 0; i < S; ++i) sum += M[(size_t) j * S + i] * qt[i];
                        ref[(size_t) j + (size_t) S * (h + (size_t) H * (t + (size_t) n_tokens * ns))] = sum * scale;
                    }
                }
                for (int j = 0; j < S; ++j) {
                    for (int i = 0; i < S; ++i) {
                        ref[(size_t) j + (size_t) S * (h + (size_t) H * ((size_t) n_tokens + i))] =
                            M[(size_t) j * S + i];
                    }
                }
            }
        }

        // note: the output layout is [S*H, n_tokens + S] with the state at the end.
        // ref above is [S, H, n_tokens+S] so compare through the same index.
        double max_diff = 0.0;
        double max_attn = 0.0;
        double max_state = 0.0;
        for (int h = 0; h < H; ++h) {
            for (int j = 0; j < S; ++j) {
                for (int t = 0; t < n_tokens; ++t) {
                    const double a = got[(size_t) j + (size_t) S * (h + (size_t) H * t)];
                    const double b = ref[(size_t) j + (size_t) S * (h + (size_t) H * t)];
                    max_diff = std::max(max_diff, fabs(a - b));
                    max_attn = std::max(max_attn, fabs(a - b));
                }
                for (int i = 0; i < S; ++i) {
                    const double a = got[(size_t) S * H * n_tokens + (size_t) h * S * S + (size_t) j * S + i];
                    const double b = ref[(size_t) j + (size_t) S * (h + (size_t) H * (n_tokens + i))];
                    max_diff = std::max(max_diff, fabs(a - b));
                    max_state = std::max(max_state, fabs(a - b));
                }
            }
        }
        printf("n_tokens=%3d max diff %.6g (attn %.6g state %.6g)\n", n_tokens, max_diff, max_attn, max_state);
        if (max_diff > 1e-4) fails++;

        ggml_backend_buffer_free(buf);
        ggml_backend_free(be);
        ggml_free(ctx);
    }

    if (fails) {
        printf("FAILED (%d)\n", fails);
        return 1;
    }
    printf("PASSED\n");
    return 0;
}
