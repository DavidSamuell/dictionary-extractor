#!/usr/bin/env bash
#
# One-shot local migration to the experiment-aware stage-1 / stage-2 layout.
#
# Before:
#   <lang>/outputs/stage-1/<stem>/
#       <stem>_stage1.tsv         (prediction)
#       <stem>_stage1_GOLD.tsv    (gold)
#       <stem>_stage1_raw.json
#       <stem>_stage1_input.json
#       <stem>_usage.json
#   <lang>/outputs/stage-2/<stem>/
#       <stem>.tsv  + .json  + *_usage.json
#
# After:
#   <lang>/outputs/stage-1-gold/<stem>/<stem>_stage1_GOLD.tsv
#   <lang>/outputs/stage-1/legacy/<stem>/<stem>_stage1.tsv + ... json files
#   <lang>/outputs/stage-2/legacy/<stem>/...
#
# `assets/` is gitignored so this rearrangement is local-only.  Run once
# from the repo root, then delete this script if you like.
#
# Pass a custom samples root as the first arg; defaults to
# `assets/dictionaries`.

set -euo pipefail

ROOT="${1:-assets/dictionaries}"

if [ ! -d "$ROOT" ]; then
    echo "error: $ROOT is not a directory" >&2
    exit 1
fi

# Find every <lang>/outputs that has a stage-1/ subdir.  Glob handles both
# `assets/dictionaries/samples/<lang>/outputs/` and
# `assets/dictionaries/samples-2/<lang>/outputs/` shapes.
shopt -s nullglob
for LANG_OUT in "$ROOT"/*/*/outputs "$ROOT"/*/outputs; do
    [ -d "$LANG_OUT/stage-1" ] || [ -d "$LANG_OUT/stage-2" ] || continue

    # ----- Stage 1: lift gold to sibling stage-1-gold/, predictions to legacy/ -----
    if [ -d "$LANG_OUT/stage-1" ] && [ ! -d "$LANG_OUT/stage-1/legacy" ]; then
        GOLD_ROOT="$LANG_OUT/stage-1-gold"
        LEGACY_ROOT="$LANG_OUT/stage-1/legacy"

        mkdir -p "$GOLD_ROOT" "$LEGACY_ROOT"
        moved_gold=0
        moved_pred=0

        for STEM_DIR in "$LANG_OUT"/stage-1/*/; do
            STEM="$(basename "$STEM_DIR")"
            [ "$STEM" = "legacy" ] && continue
            # If this looks like an already-experiment slot (run_config.json
            # next to per-stem subdirs, not _stage1.tsv directly), skip it.
            if [ -f "$STEM_DIR/run_config.json" ]; then
                echo "  [skip] stage-1 already an experiment slot: $STEM_DIR"
                continue
            fi

            GOLD_FILE="$STEM_DIR/${STEM}_stage1_GOLD.tsv"
            if [ -f "$GOLD_FILE" ]; then
                mkdir -p "$GOLD_ROOT/$STEM"
                mv "$GOLD_FILE" "$GOLD_ROOT/$STEM/"
                moved_gold=$((moved_gold + 1))
            fi

            mv "$STEM_DIR" "$LEGACY_ROOT/$STEM"
            moved_pred=$((moved_pred + 1))
        done

        echo "[ok] $LANG_OUT/stage-1: ${moved_gold} gold → stage-1-gold/, ${moved_pred} stem dirs → stage-1/legacy/"
    elif [ -d "$LANG_OUT/stage-1/legacy" ]; then
        echo "[skip] stage-1 already migrated: $LANG_OUT/stage-1"
    fi

    # ----- Stage 2: lift <stem>/ directly under stage-2/ into stage-2/legacy/ -----
    if [ -d "$LANG_OUT/stage-2" ] && [ ! -d "$LANG_OUT/stage-2/legacy" ]; then
        S2_LEGACY="$LANG_OUT/stage-2/legacy"
        mkdir -p "$S2_LEGACY"
        moved_s2=0

        for STEM_DIR in "$LANG_OUT"/stage-2/*/; do
            STEM="$(basename "$STEM_DIR")"
            [ "$STEM" = "legacy" ] && continue
            if [ -f "$STEM_DIR/run_config.json" ]; then
                echo "  [skip] stage-2 already an experiment slot: $STEM_DIR"
                continue
            fi
            mv "$STEM_DIR" "$S2_LEGACY/$STEM"
            moved_s2=$((moved_s2 + 1))
        done

        echo "[ok] $LANG_OUT/stage-2: ${moved_s2} stem dirs → stage-2/legacy/"
    elif [ -d "$LANG_OUT/stage-2/legacy" ]; then
        echo "[skip] stage-2 already migrated: $LANG_OUT/stage-2"
    fi
done

echo
echo "Done.  Next steps:"
echo "  1. uv run dictextractor-extract --strategy two_stage --stage 1 \\"
echo "         --samples-dir <root> --experiment-name <name> ..."
echo "  2. uv run dictextractor-eval-s1 --samples-dir <root> --all-experiments"
