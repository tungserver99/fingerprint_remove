#!/usr/bin/env bash
set -euo pipefail

python run_ccdr_stream.py

python run_ccdr_evals_with_fsr.py \
  --model-path outputs/ccdr_if \
  --output-dir outputs/ccdr_if/results \
  --seqlen 2048 \
  --dtype bf16 \
  --model-fingerprint-dir Model-Fingerprint \
  --fingerprint-data Model-Fingerprint/dataset/llama_fingerprint_chat \
  --fingerprint-filename ccdr_publish \
  --template instruction_attack
