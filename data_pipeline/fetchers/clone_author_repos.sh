#!/usr/bin/env bash
# =============================================================================
# clone_author_repos.sh
# =============================================================================
# Clone the academic AAD model repositories into external_libs/ so that the
# Neurophile adapter shim layers can import the real model classes.
#
# After cloning, follow the TODO instructions in:
#   src/neurophile/models/adapters/kul_cnn_adapter.py
#   src/neurophile/models/adapters/mesgarani_crn_adapter.py
#
# Usage:
#   bash data_pipeline/fetchers/clone_author_repos.sh
#   bash data_pipeline/fetchers/clone_author_repos.sh --update   # git pull existing
#
# Requirements: git
# =============================================================================

set -euo pipefail

EXTERNAL_DIR="external_libs"
UPDATE_MODE=false

# ── Parse arguments ────────────────────────────────────────────────────────────
for arg in "$@"; do
  case $arg in
    --update) UPDATE_MODE=true ;;
    --help)
      echo "Usage: bash clone_author_repos.sh [--update]"
      echo "  --update : git pull all already-cloned repos instead of cloning fresh"
      exit 0
      ;;
  esac
done

mkdir -p "$EXTERNAL_DIR"

# ── Helper functions ───────────────────────────────────────────────────────────
clone_or_update() {
  local name="$1"
  local url="$2"
  local dest="${EXTERNAL_DIR}/${name}"

  if [ -d "$dest/.git" ]; then
    if [ "$UPDATE_MODE" = true ]; then
      echo "  ↻ Updating ${name} …"
      git -C "$dest" pull --ff-only
    else
      echo "  ✓ ${name} already cloned (use --update to refresh)"
    fi
  else
    echo "  ↓ Cloning ${name} …"
    git clone --depth 1 "$url" "$dest"
    echo "  ✓ ${name} cloned"
  fi
}

echo ""
echo "============================================================"
echo "  Neurophile — Cloning External AAD Model Repositories"
echo "  Destination: $(pwd)/${EXTERNAL_DIR}/"
echo "============================================================"
echo ""

# ── KU Leuven CNN ─────────────────────────────────────────────────────────────
# Vandecappelle et al. (2021) — EEG-based locus of auditory attention CNN
# After cloning, check: external_libs/locus-of-auditory-attention-cnn/
#   └── Look for the main model class (typically in model.py or src/)
#   └── Update kul_cnn_adapter.py with the exact import path
echo "1/3  KU Leuven Locus-of-Auditory-Attention CNN"
clone_or_update \
  "locus-of-auditory-attention-cnn" \
  "https://github.com/exporl/locus-of-auditory-attention-cnn.git"

# ── Naplib-Python (Mesgarani Lab tools) ───────────────────────────────────────
# Mesgarani Lab's official Python toolbox for neural-acoustic processing.
# After cloning, check: external_libs/naplib-python/naplib/
#   └── Look for model classes related to CRN or neural tracking
#   └── Update mesgarani_crn_adapter.py with the exact import path
echo ""
echo "2/3  Mesgarani Lab — naplib-python"
clone_or_update \
  "naplib-python" \
  "https://github.com/naplab/naplib-python.git"

# ── Auditory Cortex Attention (supplementary Mesgarani models) ────────────────
# Additional neural tracking models from the Mesgarani group
echo ""
echo "3/3  EXPORL AAD Utilities (KU Leuven supplementary tools)"
clone_or_update \
  "auditory-eeg-challenge" \
  "https://github.com/exporl/auditory-eeg-challenge-2023-code.git"

# ── Post-clone instructions ────────────────────────────────────────────────────
echo ""
echo "============================================================"
echo "  All repositories cloned to ./${EXTERNAL_DIR}/"
echo "============================================================"
echo ""
echo "  NEXT STEPS:"
echo "  1. Inspect the cloned repos to find the model class entry points:"
echo "     ls ${EXTERNAL_DIR}/locus-of-auditory-attention-cnn/"
echo "     ls ${EXTERNAL_DIR}/naplib-python/naplib/"
echo ""
echo "  2. Update the adapter shim layers with the real import paths:"
echo "     src/neurophile/models/adapters/kul_cnn_adapter.py"
echo "     src/neurophile/models/adapters/mesgarani_crn_adapter.py"
echo "     (Follow the '# TODO (implementer)' comments in each file)"
echo ""
echo "  3. Add external_libs/ to sys.path or install as editable packages:"
echo "     pip install -e ${EXTERNAL_DIR}/naplib-python/"
echo "     export PYTHONPATH=\$(pwd)/${EXTERNAL_DIR}:\$PYTHONPATH"
echo ""
echo "  4. Set use_external=True when instantiating adapters:"
echo "     KULAdapter(num_eeg_channels=64, use_external=True)"
echo ""
