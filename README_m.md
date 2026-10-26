# Gated DeltaNet + NSA in qvac-fabric-llm.cpp

Native ggml/GGUF support for `Idiap/gated-deltanet-nsa-0.4B-10B`.
The implementation includes conversion of Hugging Face (HF) checkpoints to
GGUF, loading of GGUF files, CPU and GPU inference, prompt processing,
token-by-token generation, validation and automated tests. Inference uses C++
only. Python is used for conversion and for the reference check. The K/V cache
holds attention keys and values; q8_0 stores them in 8-bit blocks. f32 and f16
are 32-bit and 16-bit floating-point formats.

GGUF is the model file format used by llama.cpp, and ggml is its C inference
library. Metal is Apple's GPU API; Vulkan is a cross-platform GPU API.

## Overview

| Area | Implementation |
|---|---|
| Native inference | HF checkpoint to GGUF; C++/ggml graph; Python only for conversion and reference |
| CPU | Full model; f32, f16 and q8_0 weights; f16 and q8_0 KV cache |
| GPU | Metal (main): f32, f16 and q8_0 K/V. Vulkan (second): f32 K/V |
| Generation | Prompt processing and token-by-token decoding |
| Validation | Operation and backend tests, end-to-end tests, and a PyTorch reference |
| Quantization | f16 and q8_0 weights; q8_0 KV cache on CPU and Metal |

## Implemented components

The implementation has these parts:

- `conversion/hybrid_gated_deltanet.py` - Hugging Face to GGUF converter.
- `src/models/hybrid-gated-deltanet.cpp` - model load and graph.
- `src/llama-arch.*`, `src/llama-model.cpp`, `src/llama-context.cpp` - model
  registration and hybrid memory setup.
- `src/llama-memory-nsa-summary.*` - one block-summary buffer per layer.
- `src/llama-memory-hybrid-nsa.*` - hybrid memory that uses the summary buffer.
- Four ggml operations:
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

## Model

- 24 layers: even layers are Gated DeltaNet (GDN), odd layers are sparse
  attention (NSA).
- 12 GDN + 12 NSA.
- 32 query heads, 2 key/value heads, head size 64 for NSA.
- 8 GDN heads, head size 128 for GDN.
- block size 64, top-k 16 (the selector keeps the best 16 blocks), local window
  512.
- 358M parameters, 1.33 GiB in f32, context 4096.
- Base language model, not a chat model. Completion is the intended mode.


## Validation methodology and results

The official `fla` (flash-linear-attention) implementation needs Triton and a
CUDA GPU. This work was done on macOS, where those kernels cannot run.
`tests/nsa_reference.py` is a slow PyTorch version of the same math:

- GDN (Gated DeltaNet) recurrence;
- RoPE (rotary position encoding) with split-half layout;
- mean-pooled compressed K/V;
- block selection with log-sum-exp (lse) and forced first/current/previous
  blocks;
- selected attention with a mask that blocks future tokens;
- sliding window;
- sigmoid gates.

The Python version runs separately from the C++ code. It reads the same
Hugging Face checkpoint and produces logits for the same prompt. The C++ test
writes logits for every token. The Python script compares all logits and the
top-1 token. This checks the model math and graph, not just the generated text.

Measured results:

`top-1` is the highest-scoring next token. `max` and `mean` are the largest and
average differences in the output scores (logits); `max abs diff` is the largest
absolute difference.

| Check | Result |
|---|---|
| GDN operation, 1..512 tokens | max abs diff 4.1e-10 vs simple recurrence |
| NSA operations, 128 tokens, CPU | max abs diff 2.1e-7 |
| NSA operations, 128 tokens, Metal | max abs diff 2.4e-7 |
| NSA operations, 128 tokens, Vulkan | max abs diff 2.4e-7 |
| Model vs reference, 23 tokens, CPU | max 0.00166, mean 0.00016, top-1 23/23 |
| Model vs reference, 98 tokens, CPU | max 0.00496, mean 0.00023, top-1 98/98 |
| Model vs reference, 23 tokens, Metal | max 0.0101, mean 0.00080, top-1 23/23 |
| Model vs reference, 23 tokens, Vulkan | max 0.00804, mean 0.00077, top-1 23/23 |
| Model vs reference, 23 tokens, CPU, F16 weights | max 0.0691, mean 0.00547, top-1 23/23 |
| Model vs reference, 23 tokens, CPU, Q8_0 weights | max 0.326, mean 0.0214, top-1 23/23 |
| Model vs reference, 23 tokens, CPU, Q8_0 KV | max 0.0483, mean 0.00417, top-1 23/23 |
| Model vs reference, 23 tokens, Metal, Q8_0 KV | max 0.0408, mean 0.00419, top-1 23/23 |
| Prompt vs generation, CPU 23/98/512 | max 9.8e-4 / 1.4e-3 / 1.4e-3, top-1 0 |
| Prompt vs generation, CPU Q8_0 KV 23/512 | max 0.0082 / 0.0255, top-1 0 |
| Prompt vs generation, Metal 23/98/512 | max 0.0088 / 0.0088 / 0.0129, top-1 0 |
| Prompt vs generation, Metal Q8_0 KV 23/98/512 | max 0.0321 / 0.0321 / 0.0399, top-1 0 |
| Prompt vs generation, Vulkan 23/98 | max 0.0080 / 0.0094, top-1 0/23 and 1/98 |

Tolerances:

- operation tests: 1e-5 on CPU, 1e-4 on GPU;
- model: top-1 must match. With f32 weights, the largest CPU logit difference
  should stay below 0.005 for short contexts, and below 0.01 on GPU;
- tests with F16/Q8_0 weights or Q8_0 KV use a larger limit of 0.1 for the
  largest logit difference. Top-1 still must match; GPU long runs allow one
  mismatch, CPU allows none.

Differences come from f32/f16 rounding and from different summation order on
each backend. The checks compare numbers and top-1 tokens; identical bits are
not expected.

## Performance

Apple M4, 6 CPU threads. f32, f16 and q8_0 weights.

Short context: 128-token prompt (`pp128`), 32 generated tokens (`tg32`).

| backend | pp128 | tg32 |
|---|---:|---:|
| CPU f32 | 441 t/s | 63.0 t/s |
| Metal f32 | 2062 t/s | 56.9 t/s |
| Vulkan f32 | 188 t/s | 20.2 t/s |
| CPU f16 | 315 t/s | 104.6 t/s |
| Metal f16 | 2182 t/s | 98.5 t/s |
| CPU q8_0 | 534 t/s | 74.8 t/s |
| Metal q8_0 | 1937 t/s | 95.2 t/s |
| CPU f16 + Q8_0 KV | 553 t/s | 89.7 t/s |
| Metal f16 + Q8_0 KV | 2034 t/s | 97.2 t/s |

Long context (98-token prompt, 4096 context):

| backend | prompt | decode |
|---|---:|---:|
| Metal | 1362 t/s | 43.6 t/s |
| Vulkan | 153 t/s | 17.0 t/s |

Memory: about 1.5 GB of RAM for the f32 model plus runtime. The summary cache is
small: about 0.8 MB for 12 NSA layers at 4096 context. F16 weights use 685 MiB,
and Q8_0 weights use 364 MiB. Both give coherent output on CPU and Metal. The
smaller weights reduce the data read for each generated token, which makes
decode faster.

Metal is much faster than CPU for prompt processing. Decode is close to CPU
because it is memory-bound and the model is small. Vulkan through MoltenVK is
the slowest path. It is used to check that the code works on another GPU API and
to compare results.

## Design and implementation notes

- The graph gets its tensor shapes from the model parameters.
  `load_arch_hparams` checks the expected values. An incompatible checkpoint
  stops with an error at load time; it does not build a graph with wrong
  dimensions.
- GDN uses `build_delta_net_fused` and `GGML_OP_GATED_DELTA_NET`. The existing
  GDN kernel is used without changes.
- The four NSA operations create block summaries, store new summaries, select
  blocks, and run selected and sliding-window attention. CPU, Metal and Vulkan
  each provide the needed kernels or shaders.
- The summary cache stores the mean K and V values for each completed 64-token
  block. `NSA_SUMMARY_STORE` reads `nsa_range` and writes only new blocks.
  Completed blocks are not pooled again on every graph. The buffer is saved and
  loaded with the context state.
- Metal attention and summary-store use groups of 32 GPU threads. Selection
  also uses a 32-thread group when there are at least 32 blocks. Each thread
  scans part of the blocks and keeps its own short list of the best block
  scores. Thread 0 then merges the 32 lists. With fewer than 32 blocks (under
  2048 tokens at block size 64), most threads are idle. The writes to shared GPU
  memory, the synchronization point and the final merge then cost more than one
  single-thread loop. Those cases use the single-thread path. Vulkan uses the
  same threshold.
- Metal q8_0 K/V is converted back to float inside the NSA kernels. Each row is
  read as q8_0 blocks and converted before the dot product or sum. This keeps
  the cache small and avoids a separate pass over the whole cache. CPU uses the
  ggml type traits for the same conversion; Vulkan uses f32 K/V.
- The tokenizer is a byte-pair encoding (BPE) tokenizer based on sentencepiece.
  The converter writes `tokenizer.ggml.add_space_prefix=true`, and the BPE code
  in `src/llama-vocab.cpp` uses it. Token IDs match the HF `LlamaTokenizer` on
  the tested strings, including English, contractions, numbers and Cyrillic.
- The model is a base language model for completion; it has no chat template.
  An instruct checkpoint would need its own chat template.

## Status

Done:

- Hugging Face checkpoint to GGUF conversion.
- GGUF loading and native C++ inference.
- CPU inference with f32, f16 and q8_0 weights.
- Metal inference. Metal is the main GPU path; it supports f32, f16 and q8_0
  K/V.
- Vulkan inference. Vulkan is the second GPU path; it supports f32 K/V.
- Summary cache for NSA blocks; the state can be saved and loaded.
- f16 and q8_0 weights tested on CPU and Metal.
- q8_0 KV cache on CPU and Metal.
- Operation tests, backend tests, end-to-end tests, tokenizer checks and
  reference checks.

Not done:

- CUDA and mobile: this work does not include them. The work was done on macOS,
  which has no CUDA device, and the model is a desktop/server target. Metal is
  the main GPU path.
- F16 and q8_0 K/V on Vulkan: Vulkan supports f32 K/V only. Metal already covers
  the f16 and q8_0 GPU paths. The f32 Vulkan path is enough to check that the
  code works on another GPU API and to compare results. Adding f16 or q8_0
  would need more shader versions and an f16 mask path.
- Q4/Q5 KV cache on Metal: Metal supports Q8_0 K/V. Q4 and Q5 would need more
  conversion code in the Metal kernels. CPU can read them through the ggml type
  traits.
- Multi-sequence and batched prompt processing: the model runs one sequence at
  a time. Its summary cache belongs to one sequence and only adds new blocks.
  Several sequences would need separate summary state and synchronization. The
  graph checks `n_seqs == 1` and stops with a clear error for other values.
- Context rollback and shifting: the summary cache keeps completed blocks and
  only adds new ones. Rollback would need a way to remove or mark old summary
  entries invalid. `seq_rm` returns false; `seq_cp`, `seq_keep`, `seq_add` and
  `seq_div` stop with a clear message.
- Full `ctest` and `test-llama-archs`: the targeted tests cover the
  model-specific code. The full suite is much larger and includes many
  unrelated targets; the tests used here can be rerun with
  `tests/run-nsa-checks.sh`.
- Official `fla` comparison: the official kernels need Triton and a CUDA GPU.
  This work was done on macOS, where they cannot run. The PyTorch version uses
  the same math and gives an independent check.

## Summary

The implementation provides a complete native path: Hugging Face to GGUF
conversion, GGUF loading, CPU inference, Metal inference, Vulkan inference, a
summary cache for NSA blocks, f16/q8_0 weights, q8_0 KV on CPU and Metal, prompt
processing, token-by-token generation, operation tests, prompt and generation
tests and a PyTorch reference comparison. The NSA operations pass the project
`test-backend-ops` cases on Metal and Vulkan. The main CPU and Metal paths are
complete. Vulkan adds a second GPU path with f32 K/V. The model is a base
language model; completion is the intended mode.

## Possible extensions

Optional extensions:

1. Add f16 and q8_0 K/V support to Vulkan to make the second GPU path more
   complete.
2. Tune the selection threshold and speed up the single-thread short-context
   path.
3. Add support for several sequences, rollback and context shifting for server
   use.
4. Add the model to `test-llama-archs` and run the full `ctest`.
5. Add CUDA support and run the official `fla` comparison when CUDA hardware is
   available.

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

Build only `llama-completion` for a run build. A normal
`cmake --build build -j` also builds the test targets. If tests are enabled,
`test-chat` needs the `tools/mtmd` include path when
`LLAMA_BUILD_SERVER=OFF`; the `all` target can then stop before linking the
tools.

Vulkan on macOS uses MoltenVK:

```bash
brew install molten-vk vulkan-headers vulkan-loader shaderc glslang vulkan-tools

cmake -S . -B build-vk \
  -DCMAKE_BUILD_TYPE=Release \
  -DGGML_VULKAN=ON -DGGML_METAL=OFF \
  -DLLAMA_BUILD_TOOLS=ON -DLLAMA_BUILD_TESTS=OFF \
  -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_SERVER=OFF
cmake --build build-vk --target llama-completion -j 8
```

The first Vulkan build takes a few minutes. Most of the time is spent compiling
one large file, `ggml-vulkan.cpp`, and generating shaders. The build prints
nothing during this step and can look stuck. Check CPU/Activity Monitor; expect
3-7 minutes on an M-series Mac. The build is safe to leave running.

## Convert

```bash
uv run --project . python convert_hf_to_gguf.py \
  gated-deltanet-nsa-0.4B-10B \
  --outtype f32 \
  --outfile gated-deltanet-nsa-0.4B-10B/gated-deltanet-nsa-0.4B-10B-f32.gguf
```

Place the Hugging Face checkpoint in the `gated-deltanet-nsa-0.4B-10B/` directory
(gitignored). The f32 GGUF is about 1.33 GiB.

F16 and Q8_0 model files:

```bash
uv run --project . python convert_hf_to_gguf.py \
  gated-deltanet-nsa-0.4B-10B \
  --outtype f16 \
  --outfile gated-deltanet-nsa-0.4B-10B/gated-deltanet-nsa-0.4B-10B-f16.gguf

build/bin/llama-quantize \
  gated-deltanet-nsa-0.4B-10B/gated-deltanet-nsa-0.4B-10B-f32.gguf \
  gated-deltanet-nsa-0.4B-10B/gated-deltanet-nsa-0.4B-10B-q8_0.gguf Q8_0
```

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

`-ot` also puts the small input embedding on Metal. Without it, llama.cpp keeps
the embedding lookup on CPU/BLAS. This is normal llama.cpp placement; the NSA
operations still run on Metal.

Q8_0 KV cache on Metal:

```bash
build/bin/llama-completion \
  -m gated-deltanet-nsa-0.4B-10B/gated-deltanet-nsa-0.4B-10B-f16.gguf \
  -p "The capital of France is" -n 32 -c 256 -ngl 99 \
  -ctk q8_0 -ctv q8_0 -fa on \
  -ot token_embd.weight=MTL0
```

llama.cpp requires `-fa on` when the V cache is quantized. The NSA kernels do
not use flash attention. The flag also makes the mask f16, which the kernels
support. With `-ngl 99` and `-ot`, the run puts all 25 layers on `MTL0` and
reports `graph splits = 1`. The q8_0 KV path runs entirely on the GPU.

Vulkan:

```bash
build-vk/bin/llama-completion \
  -m gated-deltanet-nsa-0.4B-10B/gated-deltanet-nsa-0.4B-10B-f32.gguf \
  -p "The capital of France is" -n 32 -c 256 -ngl 99 \
  -ctk f32 -ctv f32 -fa off \
  -ot token_embd.weight=Vulkan0
```

The Vulkan NSA shaders support f32 K/V and an f32 mask. `-fa off` keeps the mask
f32; `-ctk f32 -ctv f32` keeps the cache f32. With the default f16 cache,
Vulkan would run these operations on CPU. Use the f32 flags for a GPU-only
Vulkan run.

CPU example:

```text
The capital of France is Paris. It is located in the north of the country.
```

## Tests

When the run build uses `LLAMA_BUILD_TESTS=OFF`, configure a test build with
tests enabled and build only the test targets:

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
NSA_TEST_GPU_LAYERS=99 NSA_TEST_KV_Q8=1 NSA_TEST_MODEL=... build/bin/test-nsa-model /tmp/nsa_gpu_q8kv.bin
build/bin/test-backend-ops -o NSA_ATTENTION
build/bin/test-backend-ops -o NSA_BLOCK_SUMMARY
NSA_TOKENIZE=build/bin/llama-tokenize NSA_GGUF=... NSA_MODEL_DIR=... \
  uv run --project . python tests/test-nsa-tokenizer.py
```

`test-gdn-op` compares the Gated DeltaNet operation with a simple recurrence for
1, 64, 128, 192, 193, 256 and 512 tokens.

`test-nsa-ops` builds a 128-token case with a mask that blocks future tokens. It
checks that selection never picks a future block, and compares selection and the
three attention branches with a small C++ reference.

`test-nsa-model` runs the whole model and compares prompt processing with
token-by-token generation. It skips with return code 77 when `NSA_TEST_MODEL`
is not set.

`test-backend-ops -o NSA_ATTENTION` runs the selection and attention graph on
all backends. It includes a 2048-token case that uses the 32-thread selection
kernel. The `-o NSA_BLOCK_SUMMARY` case tests the summary operation.
`test-nsa-tokenizer.py` compares the GGUF tokenizer IDs with the Hugging Face
`LlamaTokenizer` for English text, contractions, numbers and Cyrillic.

Test results:

- `test-gdn-op`: passed, 1..512 tokens.
- `test-nsa-ops`: passed on CPU, Metal and Vulkan; the summary-store check shows
  no difference.
- `test-nsa-model`: passed on CPU, Metal and Vulkan; also passed with F16/Q8_0
  weights and with Q8_0 KV on CPU and Metal (23, 98 and 512 tokens).
- `test-backend-ops -o NSA_ATTENTION`: 4/4 passed on Metal and Vulkan.
- `test-backend-ops -o NSA_BLOCK_SUMMARY`: 2/2 passed on Metal and Vulkan.
- tokenizer check: passed.

`tests/run-nsa-checks.sh` runs the CPU/Metal checks and the reference compare.
Set `NSA_VULKAN_BUILD` to also run the Vulkan checks.