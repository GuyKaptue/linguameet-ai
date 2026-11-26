#!/usr/bin/env bash
# Convert NLLB-200 (distilled 600M by default) to an int8 CTranslate2 model.
# Flags verified with `ct2-transformers-converter --help` (CTranslate2 4.9.0).
# Needs: pip install ctranslate2 transformers torch sentencepiece   (and network access to the Hugging Face Hub)
# NOTE: the NLLB weights are CC-BY-NC-4.0 (non-commercial).
set -euo pipefail
MODEL="${1:-facebook/nllb-200-distilled-600M}"
OUT="${2:-models/nllb-200-distilled-600M-ct2}"
QUANT="${3:-int8}"
ct2-transformers-converter --model "$MODEL" --quantization "$QUANT" --output_dir "$OUT" --force
echo "Converted $MODEL -> $OUT ($QUANT). Set NMT_MODEL_DIR=$OUT"
