# Image Caption Generator

A deep learning project that generates natural-language captions for images, with a React/Vite web app and a FastAPI backend on top of the trained models. It supports two caption decoders — a CNN-LSTM baseline and a CNN + visual-attention ("Show, Attend and Tell") decoder — plus caption translation and text-to-speech playback.

## Features

- ResNet-50 CNN encoder + LSTM decoder, with an optional attention decoder
- Greedy decoding and beam search (baseline decoder)
- BLEU-1–4 evaluation with multi-reference support
- Caption translation (`deep-translator`) and text-to-speech (`gTTS`)
- FastAPI backend (`backend/app.py`) and a React + Vite frontend (`frontend/`)
- Pytest suite covering vocabulary, BLEU scoring, decoders, and beam search

## Project Structure

```text
backend/            FastAPI captioning/translation/TTS API
frontend/            React + Vite web UI
outputs_baseline/    Metrics, charts, and sample predictions (baseline decoder)
outputs_attention/   Metrics, charts, and sample predictions (attention decoder)
scripts/             Flickr-style caption CSV converter
src/                 Models, training, inference, utils
tests/               Pytest suite
```

## Results

| Metric | Baseline | Attention |
|---|---|---|
| Best validation BLEU-4 | 0.1730 | 0.1919 |
| Test BLEU-4 | 0.1691 | 0.1767 |
| Test set | 810 images / 4,050 samples | 810 images / 4,050 samples |

Full metrics, charts, and per-image predictions are in `outputs_baseline/` and `outputs_attention/`.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate      # Windows
source .venv/bin/activate   # macOS/Linux

pip install -r requirements.txt
pip install fastapi "uvicorn[standard]" python-multipart   # for the backend

cd frontend && npm install
```

## Running

**Train:**

```bash
python src/train.py --captions <captions.csv> --images-root <images-root> --epochs 20 --decoder attention
```

**Inference:**

```bash
python src/infer.py --checkpoint outputs/best_captioner.pt --vocab outputs/vocab.json --image path/to/image.jpg
```

**Web app:**

```bash
uvicorn backend.app:app --reload --port 8000
cd frontend && npm run dev
```

## Testing

```bash
pytest
```
