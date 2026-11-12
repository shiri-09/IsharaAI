# Ishara AI

**Continuous Indian Sign Language recognition + multilingual AI dialogue.**
CodeSpark 2026 · Domain 3 (Smart Technologies) · PS 2.

Sign a full sentence to your webcam. Ishara AI recognises it, talks back through an LLM, and speaks the reply in Hindi, Kannada, Tamil and Telugu.

```
Browser (React + Vite + TS + Tailwind)
  webcam → MediaPipe Holistic (in-browser, WASM/GPU) → 285-dim keypoint frames
        ── WebSocket /ws/recognize ──►  FastAPI
                                         ├─ segmentation (hands up → hands down = one sentence)
                                         ├─ Transformer encoder → sentence + confidence
                                         ├─ LLM router: Gemini ⇄ Ollama (automatic fallback) + history
                                         ├─ translation (LLM) → hi / kn / ta / te
                                         ├─ TTS (gTTS) → mp3
                                         └─ JWT auth + SQLite (SQLAlchemy, Supabase/Postgres-ready)
```

Keypoints are extracted in the browser, so only small float arrays cross the network. No video is uploaded.

## Status (MVP)

| Area | State |
|---|---|
| Webcam → MediaPipe Holistic keypoints (pose + both hands + face subset) | ✅ |
| Data collection page (record labelled sentence takes) | ✅ |
| Transformer training (`python -m app.ml.train`, or the button in the UI) | ✅ |
| Live sentence recognition over WebSocket with auto segmentation | ✅ |
| Multi-turn dialogue (Gemini / local Ollama with fallback) | ✅ |
| Text + speech in Hindi, Kannada, Tamil, Telugu | ✅ |
| Auth + conversation history | ✅ local (SQLite). Supabase is next. |
| Docker | Dockerfile + compose written, not yet built/tested |

**Honest scope:** the MVP recognises a **closed set of whole sentences** (the ones you record on the Collect page). It works on sentence-level sequences, not static letters. The planned upgrade is gloss-level CTC decoding for open-vocabulary continuous recognition, plus a public continuous-ISL dataset.

## Run locally

Requirements: Python 3.12, Node 20+, optional [Ollama](https://ollama.com) with a model pulled.

```bash
cp .env.example .env        # add GEMINI_API_KEY and/or set LLM_PRIMARY=ollama
```

Backend:

```bash
cd backend
python -m venv .venv
.venv/Scripts/activate      # Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
pip install torch --index-url https://download.pytorch.org/whl/cu128   # RTX 50-series; use /whl/cpu for CPU-only
uvicorn app.main:app --reload --port 8000
```

Frontend (dev server proxies `/api` and `/ws` to :8000):

```bash
cd frontend
npm install
npm run dev                 # http://localhost:5173
```

Or run `npm run build` and the backend serves the built app at http://localhost:8000.

## Using it

1. **Register** and sign in.
2. **Collect & Train:** pick a sentence, press *Record take* (or Space), sign the whole sentence, then lower your hands. The take saves automatically. Aim for 20+ takes per sentence, from 2–3 people. Then press **Train**.
3. **Studio:** start the camera and sign. When you lower your hands, the sentence is recognised. If the confidence is high enough it goes to the conversation automatically. Otherwise you pick from the top 3. Replies come back in English plus the selected languages, with audio.

## Layout

```
backend/app/
  main.py            FastAPI app (also serves frontend/dist)
  config.py          settings from .env
  auth.py, models.py, db.py
  routers/           auth, chat, tts, data (sentences/samples/training), recognize (WebSocket)
  llm/               gemini.py, ollama.py, router.py (primary → fallback)
  services/          translate.py, tts.py
  ml/                features.py (keypoint layout + normalisation), model.py (Transformer),
                     dataset.py, train.py, infer.py
frontend/src/
  lib/               useHolistic.ts (camera + MediaPipe), keypoints.ts (mirror of features.py), api.ts, auth.tsx
  pages/             Login, Studio, Collect, History
data/sentences.json  sentence vocabulary; data/raw/<id>/*.npy recorded takes
models/              trained checkpoint (isl_transformer.pt)
```

## API

OpenAPI docs: http://localhost:8000/docs

## Docker

```bash
docker compose up --build   # http://localhost:8000
```

The compose file points `OLLAMA_URL` at the host machine. The container uses CPU torch.
