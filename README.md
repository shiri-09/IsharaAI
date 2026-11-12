# Ishara AI

**Continuous Indian Sign Language recognition + multimodal, multilingual AI dialogue.**
CodeSpark 2026 · Domain 3 (Smart Technologies) · PS 2.

Sign a full sentence to your webcam. Ishara AI recognises it, holds a multi-turn conversation through a
multimodal LLM, and answers in text and speech in Hindi, Kannada, Tamil and Telugu.

```
Browser (React + Vite + TS + Tailwind, Supabase JS)
  webcam → MediaPipe Holistic (in-browser) → 285-value keypoint frames ──── WebSocket /ws/recognize ───┐
  webcam filmstrip (start / middle / end of the sign) ─────────────────────── POST /api/chat ─────────┐ │
                                                                                                     ▼ ▼
FastAPI ─ segmentation (hands up → hands down = one sentence)
        ─ Conv + Transformer encoder ensemble → sentence + top-3 + confidence
        ─ LangGraph dialogue: classify intent → (interpret ambiguous sign) → respond → translate ∥ summarize
        ─ LLM router: Gemini ⇄ Ollama Qwen3.5 (vision), automatic fallback; filmstrip sent to the vision model
        ─ gTTS speech (hi / kn / ta / te / en)
        ─ Auth: Supabase Auth (token verified server-side) or local JWT fallback
        ─ History + rolling summaries: SQLAlchemy (SQLite, or Supabase Postgres via DATABASE_URL)
```

Keypoints are extracted in the browser, so only small float arrays cross the network. No video is uploaded for recognition.

## PS 2 requirements → where they live

| Requirement (problem statement) | Implementation |
|---|---|
| Webcam input, MediaPipe Holistic temporal keypoints (hands, body, face) | `frontend/src/lib/useHolistic.ts`, `keypoints.ts` (same layout in `backend/app/ml/features.py`) |
| Transformer sequence model, sentence-level (not letters) | `backend/app/ml/model.py`, `train.py`; trained on the ISL-CSLTR continuous-sentence corpus |
| Recognition accuracy on a continuous ISL sentence test set | `models/eval_report.json` (held-out recordings), shown on the Collect & Train page |
| Multimodal large model for multi-turn dialogue | `backend/app/llm/` Gemini + Ollama Qwen3.5 vision; webcam filmstrip sent with each signed turn |
| LangGraph for dialogue state | `backend/app/services/dialogue.py` (intent routing, ambiguity resolution, rolling summary memory) |
| Text + speech in ≥3 Indian languages | Hindi, Kannada, Tamil, Telugu: `services/translate.py`, `services/tts.py` |
| User authentication + conversation history | Supabase Auth (`backend/app/auth.py`, `frontend/src/lib/auth.tsx`), History page |
| FastAPI backend | `backend/app/main.py` |
| Dockerized deployment with Supabase authentication | `Dockerfile`, `docker-compose.yml` (set `AUTH_PROVIDER=supabase`) |

## Quick start (Docker)

```bash
cp .env.example .env        # fill in keys (see below)
docker compose up --build   # http://localhost:8080
```

The container reaches Ollama on the host through `host.docker.internal`. It uses CPU PyTorch; inference on the small recogniser is fast on CPU.

## Supabase authentication

1. Create a project at [supabase.com](https://supabase.com). In **Project Settings → API**, copy the **Project URL** and the **anon / publishable key**.
2. In `.env`:
   ```
   AUTH_PROVIDER=supabase
   SUPABASE_URL=https://<project-ref>.supabase.co
   SUPABASE_ANON_KEY=<anon key>
   ```
3. Optional: keep users and history in Supabase Postgres too:
   `DATABASE_URL=postgresql+psycopg://postgres.<ref>:<password>@aws-0-<region>.pooler.supabase.com:5432/postgres`
4. Restart. The login page shows "Secured by Supabase Auth". Sign-up and sign-in run through Supabase. The API verifies every access token with Supabase (`/auth/v1/user`), including the WebSocket.

If "Confirm email" is on in Supabase, new users must click the email link before signing in.

## Run locally (dev)

Requirements: Python 3.12, Node 20+, optional [Ollama](https://ollama.com).

```bash
cd backend
python -m venv .venv
.venv/Scripts/activate      # Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
pip install torch --index-url https://download.pytorch.org/whl/cu128   # RTX 50-series; /whl/cpu for CPU-only
uvicorn app.main:app --reload --port 8000
```

```bash
cd frontend
npm install
npm run dev                 # http://localhost:5173 (proxies /api and /ws to :8000)
```

## Training the recogniser

Data: **ISL-CSLTR** (Elakkiya R & Natarajan B, *ISL-CSLTR: Indian Sign Language Dataset for Continuous Sign
Language Translation and Recognition*, Mendeley Data V1, 2021, [doi:10.17632/kcmpdxky7p.1](https://doi.org/10.17632/kcmpdxky7p.1), CC BY 4.0).
It has 687 sentence videos covering ~100 ISL sentences from 7 signers.

```bash
cd backend
pip install -r requirements-data.txt
python scripts/fetch_isl_csltr.py              # ~7.6 GB of videos only, via HTTP range requests (not the full 8.9 GB zip)
python scripts/import_isl_csltr.py --workers 12  # MediaPipe Holistic keypoints → data/external/
python -m app.ml.train --epochs 300 --ensemble 5 --final
```

- **Features:** body-centred, shoulder-scaled upper-body, hand and face points; velocities; wrist-relative, size-normalised hand shapes; short tracking gaps interpolated.
- **Model:** linear projection → temporal convolutions → 4-layer Transformer encoder (CLS pooling). An ensemble of seeds is averaged.
- **Augmentation:** mirroring (left-handed signers), rotation, scale, shift, jitter, temporal crop and speed change, simulated hand dropouts.
- **Evaluation:** every recording belongs to exactly one of train / val / test, so test videos are never seen in training. Results go to `models/eval_report.json`. `--final` then retrains on all data for the deployed model.

Your own recordings from the **Collect & Train** page (`data/raw/`) are combined with the corpus automatically.

## Layout

```
backend/app/
  main.py, config.py, auth.py, db.py, models.py
  routers/       auth, chat, tts, data (sentences / samples / training), recognize (WebSocket)
  llm/           gemini.py, ollama.py, router.py (primary → fallback, multimodal)
  services/      dialogue.py (LangGraph), translate.py, tts.py
  ml/            features.py, model.py, dataset.py, train.py, infer.py
backend/scripts/ fetch_isl_csltr.py, import_isl_csltr.py
frontend/src/    lib/ (useHolistic, keypoints, api, auth), pages/ (Login, Studio, Collect, History)
data/            sentences.json, raw/ (own takes), external/ (corpus keypoints, git-ignored)
models/          isl_transformer.pt, eval_report.json
```

API docs: http://localhost:8000/docs (dev) or http://localhost:8080/docs (Docker).
