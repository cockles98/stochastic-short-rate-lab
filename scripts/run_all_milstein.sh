#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PRESETS=(baseline slow-revert fast-revert)

for preset in "${PRESETS[@]}"; do
  python -m cir.cli simulate-paths \
    --scheme milstein \
    --preset "${preset}" \
    --T 5 \
    --steps-per-year 252 \
    --n-paths 10 \
    --seed 42
done

python -m cir.cli terminal-hist \
  --scheme milstein \
  --preset baseline \
  --T 5 \
  --paths 50000 \
  --steps-per-year 252 \
  --seed 123

python -m cir.cli convergence \
  --scheme milstein \
  --preset baseline \
  --T 1 \
  --paths 50000 \
  --base-steps "52,104,208,416,832" \
  --seed 123

python -m cir.cli term-structure \
  --scheme milstein \
  --preset baseline \
  --Tmax 10 \
  --grid 40 \
  --paths 5000 \
  --steps-per-year 252 \
  --seed 777
