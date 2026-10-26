#!/usr/bin/env bash
# Reproducible checks for the hybrid Gated DeltaNet + NSA model.
#
#   NSA_MODEL=/path/to/model.gguf ./tests/run-nsa-checks.sh
#
# The script expects the model directory and the GGUF file to exist. If they
# do not, run the conversion command from README_m.md first.

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BUILD="${NSA_BUILD:-$ROOT/build}"
MODEL_DIR="${NSA_MODEL_DIR:-$ROOT/gated-deltanet-nsa-0.4B-10B}"
MODEL="${NSA_MODEL:-$MODEL_DIR/gated-deltanet-nsa-0.4B-10B-f32.gguf}"

if [[ ! -f "$MODEL" ]]; then
    echo "missing model: $MODEL" >&2
    exit 1
fi

cmake -S "$ROOT" -B "$BUILD" \
    -DCMAKE_BUILD_TYPE=Release \
    -DGGML_METAL=ON \
    -DLLAMA_BUILD_TOOLS=ON \
    -DLLAMA_BUILD_TESTS=ON \
    -DLLAMA_BUILD_EXAMPLES=OFF \
    -DLLAMA_BUILD_SERVER=OFF >/dev/null

cmake --build "$BUILD" -j "${NSA_JOBS:-8}" \
    --target test-gdn-op test-nsa-ops test-nsa-model llama-completion llama-bench llama-tokenize >/dev/null

echo "== GDN op =="
"$BUILD/bin/test-gdn-op"

echo "== NSA operators (CPU + GPU) =="
"$BUILD/bin/test-nsa-ops"

echo "== model, CPU =="
NSA_TEST_MODEL="$MODEL" "$BUILD/bin/test-nsa-model" /tmp/nsa_cpu.bin

echo "== model, Metal =="
NSA_TEST_GPU_LAYERS=99 NSA_TEST_MODEL="$MODEL" "$BUILD/bin/test-nsa-model" /tmp/nsa_gpu.bin

if [[ -n "${NSA_VULKAN_BUILD:-}" ]]; then
    echo "== operators, Vulkan =="
    "$NSA_VULKAN_BUILD/bin/test-nsa-ops"

    echo "== model, Vulkan =="
    NSA_TEST_GPU_LAYERS=99 NSA_TEST_KV_F32=1 NSA_TEST_MODEL="$MODEL" \
        "$NSA_VULKAN_BUILD/bin/test-nsa-model" /tmp/nsa_vk.bin
fi

echo "== tokenizer =="
NSA_TOKENIZE="$BUILD/bin/llama-tokenize" NSA_GGUF="$MODEL" NSA_MODEL_DIR="$MODEL_DIR" \
    uv run --project "$ROOT" python "$ROOT/tests/test-nsa-tokenizer.py"

echo "== PyTorch reference =="
uv run --project "$ROOT" python "$ROOT/tests/nsa_reference.py" \
    --model "$MODEL_DIR" \
    --prompt "The capital of France is Paris. It is known for the Eiffel Tower and the Louvre museum." \
    --compare /tmp/nsa_cpu.bin

echo "== sample output, CPU =="
"$BUILD/bin/llama-completion" -m "$MODEL" -p "The capital of France is" -n 16 -c 256 --simple-io 2>/dev/null | tail -2

echo "== sample output, Metal =="
"$BUILD/bin/llama-completion" -m "$MODEL" -p "The capital of France is" -n 16 -c 256 -ngl 99 --simple-io 2>/dev/null | tail -2

echo "all checks finished"
