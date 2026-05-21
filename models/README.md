# Model inference smoke tests

Scripts for comparing OCR / document-VLM models on a single PDF.

## Quick start

From the `dictionary-extractor` project root:

```bash
# Install base project deps (includes pymupdf for PDF rendering)
uv sync

# Run one model on page 0 of a PDF
python models/test_model_inference.py \
  --input /path/to/document.pdf \
  --models mineru \
  --output-dir models/outputs/my_run
```

By default the script:

1. Renders the selected PDF page(s) to PNG under `<output-dir>/pages/`
2. Runs each selected model sequentially (one GPU job at a time)
3. Writes per-model outputs under `<output-dir>/<model>/`
4. Writes `run_summary.json` with timings and artifact paths

### Common options

```bash
# Test page 0 only (default)
python models/test_model_inference.py -i doc.pdf --models paddleocr --pages 0

# Test multiple pages
python models/test_model_inference.py -i doc.pdf --models mineru --pages 0,1

# Full-document pipelines (MinerU CLI, PaddleOCR-VL, GLM-OCR)
python models/test_model_inference.py -i doc.pdf --models paddleocr glm-ocr --all-pages

# Run all four models (heavy — run one at a time on smaller GPUs)
python models/test_model_inference.py -i doc.pdf --models mineru paddleocr glm-ocr ovis
```

---

## Per-model installation

Each model has **conflicting dependencies** (different `transformers`, `paddlepaddle`, or `vllm` versions). Use **separate virtual environments** per model unless you know your stack is compatible.

Recommended layout:

```text
dictionary-extractor/
  .venv/                  # base project
  .venv-mineru/           # MinerU2.5-Pro
  .venv-paddleocr/        # PaddleOCR-VL-1.5
  .venv-glmocr/           # GLM-OCR (+ vLLM server)
  .venv-ovis/             # Ovis2.6-30B-A3B
```

---

### 1. MinerU2.5-Pro

**Model:** [opendatalab/MinerU2.5-Pro-2604-1.2B](https://huggingface.co/opendatalab/MinerU2.5-Pro-2604-1.2B)

**GPU:** ~1× GPU, 8–16 GB+ VRAM (transformers); faster with vLLM.

#### Option A — transformers (simplest, no extra server)

```bash
python -m venv .venv-mineru && source .venv-mineru/bin/activate
pip install -U pip
pip install "mineru-vl-utils[transformers]" torch torchvision
pip install pymupdf pyyaml   # for this test script
```

Run:

```bash
python models/test_model_inference.py -i doc.pdf --models mineru --pages 0
```

#### Option B — vLLM backend (faster)

```bash
pip install "mineru-vl-utils[vllm]"
```

Run:

```bash
python models/test_model_inference.py -i doc.pdf --models mineru --mineru-backend vllm
```

#### Option C — full MinerU CLI (native PDF, layout pipeline)

```bash
pip install -U "mineru[core]"
mineru-models-download --model_type all   # downloads layout + VL weights
```

Run:

```bash
python models/test_model_inference.py -i doc.pdf --models mineru --mineru-use-cli
# or
python models/test_model_inference.py -i doc.pdf --models mineru --all-pages
```

**Outputs:** `models/outputs/.../mineru/page_0000/output.md` (VL utils) or MinerU CLI markdown under the model output dir.

---

### 2. PaddleOCR-VL-1.5

**Model:** [PaddlePaddle/PaddleOCR-VL-1.5](https://huggingface.co/PaddlePaddle/PaddleOCR-VL-1.5)

**GPU:** 1× NVIDIA GPU with compute capability ≥ 8.5 recommended for Paddle GPU inference.

```bash
python -m venv .venv-paddleocr && source .venv-paddleocr/bin/activate
pip install -U pip

# Adjust CUDA wheel index for your CUDA version:
# https://www.paddlepaddle.org.cn/install/quick
python -m pip install paddlepaddle-gpu==3.2.1 \
  -i https://www.paddlepaddle.org.cn/packages/stable/cu126/
python -m pip install -U "paddleocr[doc-parser]"
pip install pymupdf pyyaml
```

Run (single page):

```bash
python models/test_model_inference.py -i doc.pdf --models paddleocr --pages 0
```

Run (full PDF):

```bash
python models/test_model_inference.py -i doc.pdf --models paddleocr --all-pages
```

#### Optional — vLLM server for faster inference

Terminal 1 — start server (Docker example from Paddle docs):

```bash
docker run --rm --gpus all --network host \
  ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlepaddle/paddleocr-genai-vllm-server:latest \
  paddleocr genai_server --model_name PaddleOCR-VL-1.5-0.9B \
  --host 0.0.0.0 --port 8080 --backend vllm
```

Terminal 2 — the test script uses the in-process `PaddleOCRVL()` API by default. To use the server, call the pipeline directly or extend the runner with `vl_rec_backend="vllm-server"`.

**Outputs:** JSON + Markdown + layout images under `.../paddleocr/page_0000/`.

---

### 3. GLM-OCR

**Model:** [zai-org/GLM-OCR](https://huggingface.co/zai-org/GLM-OCR)

**GPU:** ~8–16 GB VRAM (0.9B model via Transformers).

The smoke test uses **Hugging Face Transformers** locally (`AutoModelForImageTextToText`), following the [official model card](https://huggingface.co/zai-org/GLM-OCR). Requires `transformers>=5.9` (supports the `glm_ocr` architecture).

```bash
# In base .venv (see examples/install_models_venv.sh)
pip install "transformers>=5.9.0" torch torchvision
```

Run on default assets:

```bash
python models/test_model_inference.py --models glm-ocr --pages 0
```

Prompts (document parsing):

```bash
python models/test_model_inference.py --models glm-ocr \
  --glm-ocr-prompt "Text Recognition:"
# Also: "Formula Recognition:", "Table Recognition:"
```

**Outputs:** `output.md`, `output.txt`, and `result.json` under `.../glm-ocr/page_0000/`.

Settings in `models/config.yaml`.

---

### 4. Ovis2.6-30B-A3B

**Model:** [AIDC-AI/Ovis2.6-30B-A3B](https://huggingface.co/AIDC-AI/Ovis2.6-30B-A3B)

**GPU:** Large — plan for multi-GPU (`device_map="auto"`) or vLLM with `--tensor-parallel-size 4`. Not suitable for small 16 GB cards unless heavily quantized.

```bash
python -m venv .venv-ovis && source .venv-ovis/bin/activate
pip install -U pip
pip install torch==2.7.1 transformers==4.57.0 numpy pillow accelerate pymupdf
pip install flash-attn==2.8.3 --no-build-isolation   # optional but recommended
```

Run:

```bash
python models/test_model_inference.py -i doc.pdf --models ovis --pages 0
```

Multi-page:

```bash
python models/test_model_inference.py -i doc.pdf --models ovis --pages all
```

#### Optional — vLLM (multi-GPU)

```bash
pip install -U vllm
vllm serve AIDC-AI/Ovis2.6-30B-A3B --trust-remote-code --tensor-parallel-size 4
```

The test script uses **transformers** directly; use vLLM manually if you prefer an OpenAI-compatible server.

**Outputs:** `.../ovis/page_0000/output.txt`

---

## GPU tips

| Model | Typical VRAM | Notes |
|---|---|---|
| MinerU2.5-Pro | 8–16 GB | 1.2B VLM; vLLM is faster |
| PaddleOCR-VL-1.5 | 8–16 GB | Paddle GPU stack |
| GLM-OCR | 8–16 GB | Transformers (`transformers>=5.9`) |
| Ovis2.6-30B-A3B | 40 GB+ | MoE 30B total, ~3B active; multi-GPU likely |

Run **one model at a time** on limited VRAM:

```bash
for m in mineru paddleocr glm-ocr ovis; do
  python models/test_model_inference.py -i doc.pdf --models "$m" -o "models/outputs/$m"
done
```

---

## Output layout

```text
models/outputs/my_run/
  pages/
    page_0000.png
  mineru/
    page_0000/
      content.json
      output.md
  paddleocr/
    page_0000/
      *_res.json
      *.md
  glm-ocr/
    page_0000/
      result.json
  ovis/
    page_0000/
      output.txt
  run_summary.json
```

---

## Troubleshooting

| Issue | Fix |
|---|---|
| `CUDA out of memory` | Run one model at a time; use `--pages 0`; use vLLM / smaller batch |
| GLM-OCR `glm_ocr` not recognized | Upgrade: `pip install "transformers>=5.9.0"` |
| Paddle import error | Install matching `paddlepaddle-gpu` for your CUDA version |
| Ovis fails to load | Needs recent GPU + enough VRAM; try `--ovis-no-thinking` to reduce memory |
| MinerU CLI not found | `pip install -U "mineru[core]"` and run `mineru-models-download` |

---

## Files

| File | Purpose |
|---|---|
| `test_model_inference.py` | CLI entry point |
| `common.py` | PDF rendering, timing helpers |
| `runners/mineru.py` | MinerU2.5-Pro |
| `runners/paddleocr.py` | PaddleOCR-VL-1.5 |
| `runners/glmocr.py` | GLM-OCR |
| `runners/ovis.py` | Ovis2.6-30B-A3B |
