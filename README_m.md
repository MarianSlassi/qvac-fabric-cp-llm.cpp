# Gated DeltaNet + NSA in qvac-fabric-llm.cpp

Native ggml/GGUF support for `Idiap/gated-deltanet-nsa-0.4B-10B`.
Inference is plain C++; Python is only used for conversion and for the reference
check.

## What was added

- `conversion/hybrid_gated_deltanet.py` - HF to GGUF converter.
- `src/models/hybrid-gated-deltanet.cpp` - model load and graph.
- `src/llama-arch.*`, `src/llama-model.cpp`, `src/llama-context.cpp` - arch
  registration and hybrid memory wiring.
- `src/llama-memory-nsa-summary.*` - per-layer block-summary buffer.
- `src/llama-memory-hybrid-nsa.*` - hybrid memory with the summary buffer.
- Four new ggml ops:
  - `NSA_BLOCK_SUMMARY`
  - `NSA_SUMMARY_STORE`
  - `NSA_SELECT_BLOCKS`
  - `NSA_ATTENTION`
- CPU kernels in `ggml/src/ggml-cpu/ops.cpp`.
- Metal kernels in `ggml/src/ggml-metal/*`.
- Vulkan shaders and dispatch in `ggml/src/ggml-vulkan/*`.
- `tests/test-gdn-op.cpp`, `tests/test-nsa-ops.cpp`,
  `tests/test-nsa-model.cpp`, `tests/nsa_reference.py`,
  `tests/run-nsa-checks.sh`.
- `NSA_ATTENTION` and `NSA_BLOCK_SUMMARY` cases in
  `tests/test-backend-ops.cpp`.

## Status

Done:

- HF checkpoint to GGUF conversion.
- GGUF load and native C++ inference.
- CPU output.
- Metal output. Metal is the main GPU path.
- Vulkan output. Vulkan is an additional backend and needs f32 K/V.
- Incremental NSA block-summary cache.
- Q8_0 quantized weights.
- F16 weights tested on CPU and Metal.
- Targeted operator and end-to-end tests, plus a PyTorch reference check.

Not done:

- CUDA and mobile.
- F16 Vulkan shaders. The Vulkan path currently needs f32 K/V and `-fa off`.
- Multi-sequence/batched prompt processing.
- Context rollback/shifting. The code rejects these operations explicitly.
- Full `ctest` and `test-llama-archs` for this architecture.
- Official `fla` comparison on CUDA.

## Model

- 24 layers: even layers are Gated DeltaNet, odd layers are NSA.
- 12 GDN + 12 NSA.
- 32 q heads, 2 KV heads, head dim 64 for NSA.
- 8 GDN heads, head dim 128 for GDN.
- block size 64, top-k 16, local window 512.
- 358M params, 1.33 GiB in f32, context 4096.
- Base LM, not instruct/chat. Completion is the intended mode.


## Implementation Validation

Official `fla` (flash-linear-attention) needs Triton and a CUDA GPU. (Current host was Mac)
`tests/nsa_reference.py` is a slow PyTorch port of the fla math:

- GDN recurrence;
- RoPE with split-half layout;
- mean-pooled compressed K/V;
- block selection with lse and forced first/current/previous blocks;
- selected attention with causal mask;
- sliding window;
- sigmoid gates.

Measured results:

| Check | Result |
|---|---|
| GDN op, 1..512 tokens | max abs diff 4.1e-10 vs naive recurrence |
| NSA ops, 128 tokens, CPU | max abs diff 2.1e-7 |
| NSA ops, 128 tokens, Metal | max abs diff 2.4e-7 |
| NSA ops, 128 tokens, Vulkan | max abs diff 2.4e-7 |
| Model vs reference, 23 tokens, CPU | max 0.00166, mean 0.00016, top-1 23/23 |
| Model vs reference, 98 tokens, CPU | max 0.00496, mean 0.00023, top-1 98/98 |
| Model vs reference, 23 tokens, Metal | max 0.0101, mean 0.00080, top-1 23/23 |
| Model vs reference, 23 tokens, Vulkan | max 0.00804, mean 0.00077, top-1 23/23 |
| Prefill vs decode, CPU 23/98/512 | max 9.8e-4 / 1.4e-3 / 1.4e-3, top-1 0 |
| Prefill vs decode, Metal 23/98/512 | max 0.0095 / 0.0091 / 0.0127, top-1 0 |
| Prefill vs decode, Vulkan 23/98 | max 0.0080 / 0.0094, top-1 0/23 and 1/98 |

Tolerances:

- operator tests: 1e-5 on CPU, 1e-4 on GPU;
- model: top-1 must match; CPU max logit diff should stay under 0.005 for
  short contexts; GPU under 0.01;
- GPU prefill/decode allows one top-1 mismatch in long runs. CPU is strict.

The remaining differences are f32/f16 rounding and different backend
accumulation order. They are not exact bitwise checks.

## Performance

Apple M4, 6 CPU threads. f32, f16 and q8_0 weights.

Short context (`pp128`, `tg32`):

| backend | pp128 | tg32 |
|---|---:|---:|
| CPU f32 | 441 t/s | 63.0 t/s |
| Metal f32 | 2062 t/s | 56.9 t/s |
| Vulkan f32 | 188 t/s | 20.2 t/s |
| CPU f16 | 315 t/s | 104.6 t/s |
| Metal f16 | 2182 t/s | 98.5 t/s |
| CPU q8_0 | 534 t/s | 74.8 t/s |
| Metal q8_0 | 1937 t/s | 95.2 t/s |

Long context (98-token prompt, 4096 context):

| backend | prompt | decode |
|---|---:|---:|
| Metal | 1362 t/s | 43.6 t/s |
| Vulkan | 153 t/s | 17.0 t/s |

Memory: about 1.5 GB RSS for the f32 model plus runtime. The summary cache is
small (about 0.8 MB for 12 NSA layers at 4096 context). F16 weights are 685 MiB
and Q8_0 is 364 MiB; both produce coherent output on CPU and Metal. The lower
bit width reduces the weight read per generated token, which is why decode gets
faster.

Metal is much faster than CPU for prompt processing. Decode is close to CPU
because it is memory-bound and the model is small. Vulkan via MoltenVK is the
slowest path and is mainly a correctness target.

## Implementation notes

- `src/models/hybrid-gated-deltanet.cpp` derives shapes from hparams. The only
  fixed values left are validation checks in `load_arch_hparams` so another
  checkpoint is rejected instead of loaded with wrong shapes.
- GDN reuses `build_delta_net_fused` and the existing
  `GGML_OP_GATED_DELTA_NET`; the GDN kernel itself was not changed.
- Metal attention and summary-store use 32-lane threadgroups. Selection uses a
  32-lane threadgroup when there are at least 32 blocks; short contexts use the
  scalar path because the top-k merge overhead is higher there.
- Vulkan uses the same selection threshold.
- The tokenizer uses a sentencepiece-style BPE. The converter writes
  `tokenizer.ggml.add_space_prefix=true`, and the BPE path in
  `src/llama-vocab.cpp` now applies it. Token IDs match the HF
  `LlamaTokenizer` on the strings I tested, including English, contractions,
  numbers and Cyrillic.
- The model is a base LM. There is no chat template and none is needed for
  completion. An instruct checkpoint would need its own template.
- Summary cache stores mean K/V per completed 64-token block. `NSA_SUMMARY_STORE`
  takes the block range from `nsa_range` and writes only new blocks, so the
  context is not re-pooled on every graph. The buffer is append-only; rollback
  and shifting are not supported.

## Known limitations

- Vulkan NSA shaders are f32 only. Default f16 KV cache needs `-ctk f32
  -ctv f32 -fa off`.
- Metal supports f16/f32 K/V but not quantized KV cache. The CPU path can read
  quantized K/V.
- Selection uses a threadgroup only for long contexts; the short path is
  scalar. The threshold is a tuning point.
- Multi-sequence/batched prompt processing is not supported. The NSA graph
  asserts `n_seqs == 1`.
- Summary state is append-only. `seq_rm` returns false; `seq_cp`, `seq_keep`,
  `seq_add` and `seq_div` abort with a clear message. Context shifting and
  rollback are not supported.
- Full `ctest` and `test-llama-archs` for the new architecture were not run.
- The reference is a PyTorch port, not the official Triton `fla` kernels.
- No CUDA and no mobile support.
- Quantized KV cache works on CPU in the current kernels. Metal and Vulkan
  still need dequantization support in the NSA K/V kernels.

## Short assessment

The required path works: conversion, loading, CPU inference, Metal inference,
Vulkan inference, incremental summaries, Q8_0, operator tests, model
prefill/decode tests and a reference comparison. The new ops pass the project
`test-backend-ops` cases on Metal and Vulkan.

The weakest parts are the f32-only Vulkan shaders, the scalar short-context
selection path, the missing multi-sequence/rollback coverage and the lack of an
official `fla` comparison on CUDA. The model is a base LM, so completion is the
intended mode.

## What I would do next

1. Add f16 Vulkan shaders so the default cache works without `-fa off`.
2. Tune the selection threadgroup threshold and optimize the scalar short path.
3. Add the architecture to `test-llama-archs` and run a full `ctest`.
4. Test multi-sequence, state rollback and context shifting.
5. Add CUDA and run the official `fla` comparison there, if that is in scope.

## Build

CPU and Metal:

```bash
cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DGGML_METAL=ON \
  -DLLAMA_BUILD_TOOLS=ON \
  -DLLAMA_BUILD_TESTS=OFF \
  -DLLAMA_BUILD_EXAMPLES=OFF \
  -DLLAMA_BUILD_SERVER=OFF
cmake --build build --target llama-completion -j
```

Use `--target llama-completion` for a run build. Do not use plain
`cmake --build build -j` with tests enabled in this tree: the generic
`test-chat` target needs the `tools/mtmd` include path when
`LLAMA_BUILD_SERVER=OFF`, and `all` stops before linking the tools.

Vulkan (macOS uses MoltenVK):

```bash
brew install molten-vk vulkan-headers vulkan-loader shaderc glslang vulkan-tools

cmake -S . -B build-vk \
  -DCMAKE_BUILD_TYPE=Release \
  -DGGML_VULKAN=ON -DGGML_METAL=OFF \
  -DLLAMA_BUILD_TOOLS=ON -DLLAMA_BUILD_TESTS=OFF \
  -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_SERVER=OFF
cmake --build build-vk --target llama-completion -j 8
```

Note: The first Vulkan build is slow: most of the time is one large translation unit,
`ggml-vulkan.cpp`, plus shader generation. It prints nothing while clang is
compiling that file, so it can look stuck. Check CPU/Activity Monitor; expect
couple of minutes (3-7 minutes) on an M-series Mac. Do not interrupt it.

## Convert

```bash
uv run --project . python convert_hf_to_gguf.py \
  gated-deltanet-nsa-0.4B-10B \
  --outtype f32 \
  --outfile gated-deltanet-nsa-0.4B-10B/gated-deltanet-nsa-0.4B-10B-f32.gguf
```

The model directory is already present and is gitignored. The GGUF is about
1.33 GiB.

## Run

CPU:

```bash
build/bin/llama-completion \
  -m gated-deltanet-nsa-0.4B-10B/gated-deltanet-nsa-0.4B-10B-f32.gguf \
  -p "The capital of France is" -n 32 -c 256
```

Metal:

```bash
build/bin/llama-completion \
  -m gated-deltanet-nsa-0.4B-10B/gated-deltanet-nsa-0.4B-10B-f32.gguf \
  -p "The capital of France is" -n 32 -c 256 -ngl 99 \
  -ot token_embd.weight=MTL0
```

`-ot` puts the tiny input embedding on Metal too. Without it, llama.cpp keeps
that `GET_ROWS` node on CPU/BLAS. This is normal llama.cpp behavior and is not a
fallback of the NSA ops.

Vulkan:

```bash
build-vk/bin/llama-completion \
  -m gated-deltanet-nsa-0.4B-10B/gated-deltanet-nsa-0.4B-10B-f32.gguf \
  -p "The capital of France is" -n 32 -c 256 -ngl 99 \
  -ctk f32 -ctv f32 -fa off \
  -ot token_embd.weight=Vulkan0
```

The Vulkan NSA shaders are f32 K/V and f32 mask only. `-fa off` keeps the mask
f32; `-ctk f32 -ctv f32` keeps the cache f32. With the default f16 cache these
ops are not supported on Vulkan and the scheduler would use CPU.

CPU example:

```text
The capital of France is Paris. It is located in the north of the country.
```

## Tests

If the run build above used `LLAMA_BUILD_TESTS=OFF`, reconfigure with tests on
and build only the test targets:

```bash
cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DGGML_METAL=ON \
  -DLLAMA_BUILD_TOOLS=ON \
  -DLLAMA_BUILD_TESTS=ON \
  -DLLAMA_BUILD_EXAMPLES=OFF \
  -DLLAMA_BUILD_SERVER=OFF
cmake --build build --target test-gdn-op test-nsa-ops test-nsa-model test-backend-ops llama-completion -j
```

Then:

```bash
build/bin/test-gdn-op
build/bin/test-nsa-ops
NSA_TEST_MODEL=.../model-f32.gguf build/bin/test-nsa-model /tmp/nsa.bin
NSA_TEST_GPU_LAYERS=99 NSA_TEST_MODEL=... build/bin/test-nsa-model /tmp/nsa_gpu.bin
NSA_TEST_GPU_LAYERS=99 NSA_TEST_KV_F32=1 NSA_TEST_MODEL=... build/bin/test-nsa-model /tmp/nsa_gpu_f32.bin
build/bin/test-backend-ops -o NSA_ATTENTION
build/bin/test-backend-ops -o NSA_BLOCK_SUMMARY
NSA_TOKENIZE=build/bin/llama-tokenize NSA_GGUF=... NSA_MODEL_DIR=... \
  uv run --project . python tests/test-nsa-tokenizer.py
```

`test-gdn-op` compares the Gated DeltaNet op with a naive recurrence for 1, 64,
128, 192, 193, 256 and 512 tokens.

`test-nsa-ops` builds a 128-token causal case, checks that selection never
picks a future block, and compares selection plus the three attention branches
against a small C++ reference.

`test-nsa-model` runs the whole model and compares prompt processing with
token-by-token generation. It skips with return code 77 when `NSA_TEST_MODEL`
is not set.

`test-backend-ops -o NSA_ATTENTION` runs the selection + attention graph on all
backends, including a 2048-token case that exercises the threadgroup selection
kernel. `-o NSA_BLOCK_SUMMARY` covers the pooling op.
`test-nsa-tokenizer.py` compares the GGUF tokenizer IDs with the HF
`LlamaTokenizer` for English text, contractions, numbers and Cyrillic.

Current results:

- `test-gdn-op`: PASS, 1..512 tokens.
- `test-nsa-ops`: PASS on CPU/Metal/Vulkan; summary store checks 0 diff.
- `test-nsa-model`: PASS on CPU/Metal/Vulkan.
- `test-backend-ops -o NSA_ATTENTION`: 4/4 on Metal and Vulkan.
- `test-backend-ops -o NSA_BLOCK_SUMMARY`: 2/2 on Metal and Vulkan.
- tokenizer check: PASS.

`tests/run-nsa-checks.sh` runs the CPU/Metal checks and the reference compare.
Set `NSA_VULKAN_BUILD` to also run the Vulkan checks.