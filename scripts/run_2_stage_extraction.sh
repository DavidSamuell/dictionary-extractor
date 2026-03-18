#!/bin/bash

# uv run dictextractor-extract \
#     --strategy two_stage \
#     --model gemini/gemini-3-flash-preview \
#     --preprocess all \
#     --alphabet assets/dictionaries/samples/Yiddish-English/yiddish_alphabet.txt \
#     --intro assets/dictionaries/samples/Yiddish-English/introduction \
#     --ocr-text assets/dictionaries/samples/Yiddish-English/mathpix \
#     --input-image assets/dictionaries/samples/Yiddish-English/snippets \
#     --output assets/dictionaries/samples/Yiddish-English/outputs/yiddish-2-stage \

# uv run dictextractor-extract \
#     --strategy two_stage \
#     --model gemini/gemini-3-flash-preview \
#     --preprocess all \
#     --alphabet assets/dictionaries/samples/Chukchi-Russian/chukchi_alphabet.txt \
#     --intro assets/dictionaries/samples/Chukchi-Russian/introduction \
#     --ocr-text assets/dictionaries/samples/Chukchi-Russian/mathpix \
#     --input-image assets/dictionaries/samples/Chukchi-Russian/snippets \
#     --output assets/dictionaries/samples/Chukchi-Russian/outputs/2-stage \

uv run dictextractor-extract \
    --strategy two_stage \
    --model gemini/gemini-3-flash-preview \
    --preprocess all \
    --alphabet assets/dictionaries/samples/Nahuatl-French/nahuatl_classical_alphabet.txt \
    --intro assets/dictionaries/samples/Nahuatl-French/introduction \
    --ocr-text assets/dictionaries/samples/Nahuatl-French/mathpix \
    --input-image assets/dictionaries/samples/Nahuatl-French/snippets \
    --output assets/dictionaries/samples/Nahuatl-French/outputs/2-stage \