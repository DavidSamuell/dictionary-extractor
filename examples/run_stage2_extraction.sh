uv run dictextractor-extract \
    --samples-dir assets/dictionaries/samples-2 \
    --strategy two_stage \
    --stage 2 \
    --overwrite \
    --languages Chepang-English \
    --model gemini/gemini-3.1-pro-preview \
    --discover-extra-fields   