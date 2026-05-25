Your base `.venv` is ready. The smoke test does **not** use that env for the three OCR/VLM models — each needs its **own** env (conflicting deps). You have a **40 GB A100**, which is enough for MinerU, Paddle, and GLM-OCR.

The PDF is 1 page — default `--pages 0` is fine.

---

## 0. Shared variables (on the GPU node)

```bash
export PATH="${HOME}/.local/bin:${PATH}"
cd /data/projects/punim0478/davidsamuels/dictionary-extractor

PDF="assets/dictionaries/samples/Amharic-English/snippets/page_32.pdf"
OUT="models/outputs/amharic_page32_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUT"
```

---

## 1. One-time: create three model envs

Run from project root. These downloads can take a while (Hugging Face, Paddle, etc.).

### MinerU

```bash
uv venv .venv-mineru
UV_PROJECT_ENVIRONMENT=.venv-mineru uv pip install -U pip
UV_PROJECT_ENVIRONMENT=.venv-mineru uv pip install \
  "mineru-vl-utils[transformers]" torch torchvision pymupdf pyyaml
```

### PaddleOCR-VL (CUDA 12.x on your node → cu126 wheel)

```bash
uv venv .venv-paddleocr
UV_PROJECT_ENVIRONMENT=.venv-paddleocr uv pip install -U pip
UV_PROJECT_ENVIRONMENT=.venv-paddleocr uv pip install paddlepaddle-gpu==3.2.1 \
  --index-url https://www.paddlepaddle.org.cn/packages/stable/cu126/
UV_PROJECT_ENVIRONMENT=.venv-paddleocr uv pip install -U "paddleocr[doc-parser]" pymupdf pyyaml
```

### GLM-OCR (pick **one** path)

**A — Cloud API** (no local vLLM):

```bash
uv venv .venv-glmocr
UV_PROJECT_ENVIRONMENT=.venv-glmocr uv pip install "glmocr" pymupdf pyyaml
export GLM_OCR_API_KEY="your-key"   # or source ~/.config/dictextractor.env
```

**B — Self-hosted** (extra terminal on same GPU node):

```bash
uv venv .venv-glmocr
UV_PROJECT_ENVIRONMENT=.venv-glmocr uv pip install "glmocr[selfhosted]" "transformers>=5.3.0" "vllm>=0.19.0" pymupdf pyyaml

# Terminal 2 (or background):
UV_PROJECT_ENVIRONMENT=.venv-glmocr uv run vllm serve zai-org/GLM-OCR --port 8080 \
  --speculative-config '{"method": "mtp", "num_speculative_tokens": 3}' \
  --served-model-name glm-ocr
```

---

## 2. Test runs (one model at a time, same output folder)

Do **not** run all three in one `uv run` with the base `.venv` — imports will fail. Use each model’s env and the same `-o "$OUT"` so results land together.

```bash
# 1) MinerU
UV_PROJECT_ENVIRONMENT=.venv-mineru uv run python models/test_model_inference.py \
  -i "$PDF" --models mineru --pages 0 -o "$OUT"

# 2) PaddleOCR
UV_PROJECT_ENVIRONMENT=.venv-paddleocr uv run python models/test_model_inference.py \
  -i "$PDF" --models paddleocr --pages 0 -o "$OUT"

# 3) GLM-OCR — cloud:
UV_PROJECT_ENVIRONMENT=.venv-glmocr uv run python models/test_model_inference.py \
  -i "$PDF" --models glm-ocr --pages 0 -o "$OUT" \
  --glm-ocr-api-key "$GLM_OCR_API_KEY"

# 3) GLM-OCR — self-hosted (vLLM must be running on :8080):
UV_PROJECT_ENVIRONMENT=.venv-glmocr uv run python models/test_model_inference.py \
  -i "$PDF" --models glm-ocr --pages 0 -o "$OUT" \
  --glm-ocr-host localhost --glm-ocr-port 8080
```

Or loop after envs exist:

```bash
for m in mineru paddleocr glm-ocr; do
  case "$m" in
    mineru)    V=.venv-mineru;    EXTRA=() ;;
    paddleocr) V=.venv-paddleocr; EXTRA=() ;;
    glm-ocr)   V=.venv-glmocr;    EXTRA=(--glm-ocr-api-key "$GLM_OCR_API_KEY") ;;  # or host/port
  esac
  UV_PROJECT_ENVIRONMENT="$V" uv run python models/test_model_inference.py \
    -i "$PDF" --models "$m" --pages 0 -o "$OUT" "${EXTRA[@]}"
done
```

---

## 3. Check results

```bash
ls -R "$OUT"
cat "$OUT/run_summary.json"   # updated after each model run
```

Expected layout:

```text
$OUT/
  pages/page_0000.png
  mineru/page_0000/...
  paddleocr/page_0000/...
  glm-ocr/page_0000/...
  run_summary.json
```

---

## Practical order

| Step | Action |
|------|--------|
| 1 | Install `.venv-mineru` → run MinerU (quick sanity check) |
| 2 | Install `.venv-paddleocr` → run Paddle |
| 3 | Set up GLM (API **or** vLLM server) → run GLM |

---

## What you have vs what's missing

| Item | Status |
|------|--------|
| Base `uv` + `dictextractor` | Done (your terminal test) |
| PDF `page_32.pdf` | Exists, 1 page |
| `.venv-mineru` / paddle / glm | **Not created yet** — step 1 required |
| GLM-OCR | Needs API key **or** running `vllm serve` |

**Do not** run this until envs exist:

```bash
uv run python models/test_model_inference.py -i "$PDF" --models mineru paddleocr glm-ocr
```

That uses the base `.venv`, which does not include those model packages.

Start with MinerU only; once that succeeds, repeat for the others. If you want, I can add `examples/setup_and_run_all_models.sh` that automates the env creation and sequential runs for this PDF.
