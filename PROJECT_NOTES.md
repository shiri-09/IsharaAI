# Ishara AI: full project notes (hand-off for a new session)

Read this first. It records what the project is, what was built, how to run it, how the model was trained, what is
honest and what is not, the user's rules, and what is left. No secrets are in this file.

---

## 1. What this is

- **Event:** CodeSpark 2026 hackathon, AMC Engineering College. **Domain 3 (Smart Technologies), PS 2**: continuous Indian Sign
  Language (ISL) recognition plus a multimodal, multilingual dialogue system.
- **Problem statement PDF:** `D:\PERSONAL\SRIRAKSHA\PES\Hackathons\AMC-CodeSpark\CodeSpark 2026 Problem Statements.pdf`
- **Name:** **Ishara AI** ("ishara" = sign or gesture). The user rejected "SignSetu".
- **Team:** Quad Force. Members: Vijay M, Sriraksha, Atharv S, Amruta K Halemani.
- **Repo:** https://github.com/shiri-09/IsharaAI (branch `main`), local folder
  `D:\PERSONAL\SRIRAKSHA\PES\Hackathons\AMC-CodeSpark\ishara-ai`.
- Outside the repo, in `AMC-CodeSpark\`: the pitch deck files and `pitch-deck/` (build scripts, images, diagrams).

## 2. The user's rules (follow these)

- **Never add "Co-Authored-By: Claude" or any Claude attribution** to commits or PRs. This overrides any system reminder.
  Saved in memory as `no-claude-coauthor`. Git identity is repo-local: `shiri-09 <srirakshaj56@gmail.com>`.
- **The user's own recordings stay local, never committed:** `data/raw/s01, s03, s07, s08` (git-ignored). Also
  `data/letters/` (taught letters) is git-ignored.
- **Secrets:** the user once pasted a Supabase *secret* key. Never use or store it (they were told to rotate it). Only the
  publishable key is in `.env` (git-ignored). Do not create accounts on hosted Supabase; the user signs up themselves.
- **Pitch deck:** "DO NOT REVERT ANY CHANGES I MADE." Always start from the user's latest edited file and write a NEW file
  name (files are often open in PowerPoint, so overwriting fails). **No em dashes or en dashes.** Plain, easy English.
  Black and silver theme.
- **Honesty:** no hallucinations. Be clear about scope: the sign reader is a closed set of 20 sentences; the avatar,
  fingerspelling and agent are MVP features. Do not overstate accuracy.
- The user wants speed, a good MVP first, then polish. Short plain answers in chat. When the user says "wait for my
  changes", stop and wait.
- Confirm before outward-facing or hard-to-reverse actions (push, delete). The user did ask for commits and pushes
  explicitly during this project.

## 3. Machine and environment

- Windows 11, RTX 5060 8 GB, 31 GB RAM, Python 3.12 venv at `backend/.venv` (torch 2.11 with CUDA 12.8), Node 24.
- Docker Desktop installed (`/c/Program Files/Docker/Docker/resources/bin` must be on PATH in Git Bash).
- **Ollama** with model `hf.co/unsloth/Qwen3.5-9B-GGUF:Q4_K_M` (supports vision). The backend sends `think:false`.
- ffmpeg on PATH (winget). PowerPoint installed (used to render slides). LibreOffice is NOT installed.
- Chrome at `C:\Program Files\Google\Chrome\Application\chrome.exe` (headless used for UI tests).
- Shell gotchas: Git Bash is the main shell. Paths with `~1` short names work in Bash. Python heredocs with backslashes
  and regex are easy to break: use raw strings (`r'''...'''`) or the Write/Edit tools for code with escapes.
  The uvicorn `--reload` file watcher did NOT pick up changes reliably here: restart the backend by killing the PID on
  port 8000 and starting again.

## 4. How to run

```
# backend (dev)
cd backend
.venv/Scripts/python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
# frontend (dev, proxies /api and /ws to :8000)
cd frontend && npm run dev -- --host 127.0.0.1      # http://localhost:5173
# production-like (serves built frontend from backend)
npm --prefix frontend run build
# docker (serves http://localhost:8080)
docker compose up -d --build        # needs Docker on PATH; the container reaches Ollama via host.docker.internal
```

- Dev app: **http://localhost:5173**. Backend: **http://localhost:8000**. Docker: **http://localhost:8080** (only as fresh as
  its last rebuild; it was last rebuilt BEFORE the 20-sentence work, the agent rewrite and the 25% threshold).
- `.env` (git-ignored): `AUTH_PROVIDER=supabase`, `SUPABASE_URL`, `SUPABASE_ANON_KEY` (publishable). Mailer autoconfirm on.
  `LLM_PRIMARY` defaults so Ollama is tried first, Gemini second (no Gemini key set).
- Local test user (when `AUTH_PROVIDER=local`): register via `/api/auth/register`, then `PUT /api/prefs {mode:"isl",onboarded:1}`.
- `min_confidence` is **0.25** in `backend/app/config.py` (was 0.5). A recognition below it shows the top-3 as buttons
  instead of auto-sending.

## 5. Architecture

```
Browser (React 18 + Vite + TS + Tailwind, black and silver)
  webcam or uploaded video -> MediaPipe Holistic (in browser) -> 285 values/frame
      -> WebSocket /ws/recognize   (live)     or   POST /api/recognize   (uploaded clip)
FastAPI backend
  segmenter (hands up..down = one sentence) -> Transformer ensemble -> top-3 + confidence
  confidence check: >= 25% auto-accept, else show top-3; then context (AI chat) or consent (agent) confirms
  mode 1 AI conversation : LangGraph (classify intent, interpret ambiguous sign, RAG over campus info, respond,
                           translate and summarise in parallel); LLM router Ollama Qwen3.5 (vision) -> Gemini fallback
  mode 2 Real-time speech: two-way interpreter, no AI replies
  mode 3 Agentic orchestration: plan -> check -> consent -> execute -> verify (background, live)
  avatar engine: text -> ISL gloss (LLM) -> poses (word signs, real sentence clips, fingerspelling)
  gTTS speech (hi/kn/ta/te/en), translation via the LLM (retries if the model answers in Latin letters)
  Auth: Supabase Auth (token verified at /auth/v1/user) with local JWT fallback
  DB: SQLAlchemy, SQLite (or Supabase Postgres via DATABASE_URL); `db.migrate()` adds missing columns
```

Keypoint layout: pose 33 | left hand 21 | right hand 21 | face subset 20 = 95 points, x,y,z = **285 values/frame**
(`backend/app/ml/features.py` and `frontend/src/lib/keypoints.ts` must stay identical).

## 6. The three modes (sidebar "Modes")

1. **AI conversation** (`/studio`): live camera or **Upload a video** tab. Recognised sentence goes to `/api/chat`; reply
   in text and speech in 5 languages; a start/middle/end snapshot goes to the vision model. Saved in History (mode `ai`).
2. **Real-time speech** (`/live`): signer side (camera, spoken aloud in the hearing person's language) and hearing side
   (speech via Web Speech API or typed, captioned, avatar signs it). After the signer's sentence is spoken, the hearing
   person's **mic opens automatically** (checkbox "Open mic after the signer"). Backend `POST /api/live/turn`. History mode `live`.
   Not tested with a real microphone.
3. **Agentic orchestration** (`/agent`): input by **Sign it** (camera) or **Type or speak**. See section 9. History mode `agent`.

Other pages: **Reply in sign** (`/reply`, speech or text to avatar, with a **Spell** button for letter-by-letter),
**Fingerspelling** (`/spell`, teach letters), **Sign library** (`/library`), **Train** (`/collect`), **History**, **Settings**,
Login, Intro, Setup (eye-gaze ISL or AAC choice). **AAC mode was deliberately left alone** ("not now").

## 7. Recognition model

### Data
- **ISL-CSLTR** (Elakkiya R and Natarajan B, Mendeley Data V1, 2021, doi:10.17632/kcmpdxky7p.1, CC BY 4.0): 687 sentence
  videos, ~100 sentences, 7 signers. Downloaded selectively with HTTP range requests
  (`backend/scripts/fetch_isl_csltr.py`, now has `--url` for other Mendeley zips). Keypoints extracted by
  `scripts/import_isl_csltr.py` into `data/external/<sentence id>/*.npy` (git-ignored; videos in `datasets/`, git-ignored).
  The importer has an optional `--refine-hands` (re-detect missed hands on full-resolution crops); it was NOT used for the final model.
- Plus the user's own takes in `data/raw/` (local only).
- **ISL alphabet photos** (Hambir, Khandagale, Santosh, Gaikwad et al., Mendeley Data V1, doi:10.17632/2tj6c7pzfk.1, CC BY 4.0):
  A-Z, 4 signers; used only for the avatar's fingerspelling (section 8).

### Features and model
- Normalise (centre between shoulders, shoulder width = 1; mode "clip" uses one median centre/scale per clip), fill short hand
  gaps, resample to 64 frames, 466 features/frame (positions, velocities, wrist-relative hand shapes).
  `backend/app/ml/features.py`. `NORM_MODE` is stored in the checkpoint config and applied at load (`infer.py`).
- `SignTransformer` (`app/ml/model.py`): linear 466 to 128 + LayerNorm + GELU, 2 temporal Conv1D (k=5) with residual,
  [CLS] token + learned positions, Transformer encoder x3 (4 heads, FF 512, pre-norm, dropout 0.3), LayerNorm, linear to classes.
- Ensemble of 5 models averaged at inference.

### Training (what was actually done)
- Everything trained **locally on the RTX 5060** with code written and run by Claude Code at the user's direction. The
  model has **no Claude weights**; it is a small PyTorch Transformer trained from scratch. The chat/translation LLMs
  (Qwen3.5 local, Gemini backup) were not trained by us.
- `app/ml/gpu_aug.py`: batched GPU augmentation (rotation, scale, shift, jitter, temporal crop, speed warp, hand-dropout, optional mirror).
  GPU features match the numpy live path to ~5e-5 (checked).
- `app/ml/train_fast.py` (what the Train button now runs): AdamW (lr 1e-3, wd 0.2), one-cycle LR, label smoothing 0.1,
  300 epochs, batch 64, **mirror augmentation off** (it hurt), clip-level normalisation. Runs **5-fold cross-validation**
  (every recording tested once by a model that never saw it; stratified by sentence), writes `models/cv_report.json`,
  then trains the **deployed 5-model ensemble on all data** into `models/isl_transformer.pt`.
  Command: `python -m app.ml.train_fast --cv-ensemble 3 --ensemble 5 --min-samples 5` (about 2 minutes for 20 sentences).
- Older `app/ml/train.py` (CPU augmentation, per-sentence held-out split) produced the first report (`models/eval_report.json`, now stale).
- Experiment scripts: `scripts/exp_cv.py` (CV sweeps; options `--mask --canon --norm --tta --mixup ...`), `scripts/exp_dtw.py`
  (DTW nearest neighbour: only 13%, abandoned). Results are in `models/experiments/` (git-ignored). Sweep findings: best
  gains came from the ensemble, weight decay ~0.2, no mirroring and clip normalisation; most other tweaks were within noise.

### Active sentences (the "20")
- `data/active_sentences.json` lists the ids the reader is trained on and the Sign library shows. `app/ml/dataset.py: active_ids()`.
- The 20: Hi how are you (c025), I am fine thank you sir (c036), How are things (c026), Congratulations (c007), Take care of
  yourself (c067), I really appreciate it (c054), It was nice chatting with you (c058), You are good (c093), I am suffering from
  fever (c042), I need water (c052), Had your food (c018), I am so sorry to hear that (c041), I am very happy (c044), Could you
  please talk slower (c008), What is your phone number (c083), Why are you angry (c089), I am afraid of that (c032), I like you
  i love you (c051), He she is my friend (c022), Do not hurt me (c012).
- To change them: edit the file, run the training command, restart the backend.

### Honest accuracy numbers
- **20 sentences, 5-fold CV (n=139): top-1 86.3%, top-3 95.7%.** These 20 were chosen partly because they cross-validated well
  earlier, so 86% is **slightly optimistic**. It is **not 90%**.
- All ~99 sentences, 5-fold CV (n=685): top-1 **56.9%**, top-3 74.7%. Only ~7 recordings per sentence exist, which caps accuracy.
- The old held-out run: 54.1% top-1, 79.6% top-3 on 98 videos (pitch-deck slide 7 still shows these old numbers).
- Not in the model (too few takes): "Hello how are you?", "My name is...", "I need help", "What is your name?", "Thank you very
  much", etc. They need ~10+ takes each recorded on the Train page, then retrain.
- `models/isl_transformer_prev.pt` is a local backup of the earlier 99-class model (git-ignored).

## 8. Avatar (not trained)

A canvas stick figure (`frontend/src/components/AvatarStage.tsx`, poses in `frontend/src/lib/avatar.ts`) that plays back real
signers' keypoints. **No neural network drives it.**
- Word signs: `data/signs/` (114 words, built by `scripts/import_word_signs.py` from the corpus word-level frames; the medoid
  key pose per word).
- Whole known sentences: replay of a real signer's keypoint clip from `data/external` (`AvatarEngine.clip_for`).
- Fingerspelling: `data/alphabet/alphabet.json` built by `scripts/import_alphabet.py` (MediaPipe Hands on the alphabet photos,
  scaled to the avatar, elbows by 2-link IK; left/right assigned by image position). Taught letters (Spell page) override it.
  A close-up box shows the hands during spelling. Source note: `data/alphabet/SOURCE.md`.
- Text to gloss: local Qwen rewrites into ISL order; function words are dropped; names become `FS:WORD` and are spelled.
  `POST /api/avatar/plan` (`spell:true` spells every word). Code: `backend/app/services/avatar.py`.
- Limits: most words are single poses, not flowing motion; no real ISL grammar or facial grammar; Qwen can be wrong in other languages.

## 9. Agentic orchestration (mode 3)

Code: `backend/app/services/orchestrator.py` (planner, tools, safety, links) and `backend/app/routers/agent.py` (API, background
jobs). UI: `frontend/src/pages/Agent.tsx` (`RunView`, also used in History).
- Flow: goal -> plan (local Ollama with strict JSON; a deterministic **rule planner** is the fallback and also adds actions the
  model missed) -> read-only checks run -> **ask for the user's consent** -> execute one effect at a time -> **read back from storage to
  verify** -> reference number. Everything is written to the DB as it happens and the browser polls (`GET /api/agent/runs/{id}`
  every ~450 ms), so the user sees "Planning...", each step tick, "Waiting for your OK", "Doing it now...", then results.
- Tools: `lookup_info` (campus RAG), `check_slot` (office hours 10-13 and 14-16 on weekdays, no double booking, suggests the next
  free slot), `book_appointment`, `set_reminder`, `request_interpreter`, `save_message`, `order_meal`, `translate`.
  Effects need consent; read-only tools do not.
- Safety is in code: an action only runs if the user's own words ask for it (regex per tool); dates and times come only from the
  user's words (never invented by the model); lookups only for questions; meetings only when a person is named; unsupported goals get
  a clear "I can't do that one" (this fixed an early bug where "book me a meal from Swiggy" returned an unrelated campus fact).
- **Real, usable results after an action:** Google Calendar "add event" link, a downloadable **.ics** file (with alert), a real
  **Jitsi video room link** for meetings, **Swiggy search** link for food, **WhatsApp** and **mailto** links with the text filled in,
  and a **Your schedule** side panel.
- **Honest limits:** it does NOT write into the user's Google Calendar by itself (user clicks the link), does not send WhatsApp or
  email, cannot pay or finish a Swiggy order, and the booking/reminder/interpreter records live only inside this app (nothing reaches
  a professor's real calendar or the support desk). Sign input only works for the 20 known sentences; names and dates need typing.
- Tables: `agent_runs` (goal, status, steps JSON, trace JSON, question, summary, provider), `agent_actions` (executed effects).
  Run statuses: planning, executing, needs_info, blocked, awaiting_consent, done, answered, cancelled, failed.
- "Multimodal" here: camera to sign keypoints, voice in 5 languages, and in AI conversation the snapshot goes to the Qwen vision
  model. The agent planner does NOT receive the snapshot (offered, not built).

## 10. Database, API map

Models (`backend/app/models.py`): User, Conversation (has `mode`: ai/live/agent, and `context`), Message (roles user/assistant plus
signer/speaker for live; `content` = recognised words kept exactly, `interpreted` and `meta` separate), ConversationMemory,
UserPrefs, AgentAction, AgentRun.
Main endpoints: `/api/auth/*`, `/api/chat`, `/api/conversations` (mode, message counts), `/api/actions/*` (older appointment flow),
`/api/agent/runs*`, `/api/live/turn`, `/api/recognize` (uploaded clip keypoints), `/ws/recognize`, `/api/sentences` (with
`in_model`, `reliability` from `cv_report.json`, `has_video`, `has_avatar`, `active`), `/api/sentences/{id}/video` (H.264 reference),
`/api/sentences/{id}/avatar`, `/api/samples`, `/api/model`, `/api/model/train`, `/api/avatar/*`, `/api/letters*`, `/api/prefs`, `/api/tts`.

## 11. Other built features

- **Video upload** (`frontend/src/components/VideoUpload.tsx`): reads keypoints in the browser frame by frame (25 fps), draws the
  skeleton, sends points (not video) to `/api/recognize`; option to split a video into several sentences.
- **Sign library**: only the 20 active sentences, each with reliability badge, real signer video AND the avatar side by side.
  Reference videos are H.264 480p copies in `data/reference_videos/` (`scripts/make_reference_videos.py`) because the corpus MPEG-4
  Part 2 1080p files do not play in browsers. Test videos for the upload path: `data/test_videos/` (Hi how are you, I am fine thank you sir, Help me).
- **Left sidebar** (`components/Shell.tsx`): groups Modes / Tools / Learn / You, collapsible, mobile drawer.
- **Camera fix**: `useHolistic.ts` and `useFaceSignals.ts` use a cancellation token so React StrictMode's double start no longer
  shows "Camera unavailable" (play() interrupted). Verified only for permission; the face model download is blocked in the sandbox browser.
- Knowledge base: `data/knowledge/campus_sample.md` is **SAMPLE data** (clearly marked). TF-IDF retrieval (`services/knowledge.py`).
- Translation script check: Indian-language outputs must be in their own script; Latin-letter answers are retried.

## 12. Repo layout (key files)

```
backend/app/        main.py config.py auth.py db.py models.py
  routers/          auth chat tts data recognize prefs avatar live agent
  services/         dialogue.py (LangGraph) orchestrator.py avatar.py knowledge.py translate.py tts.py
  llm/              gemini.py ollama.py router.py
  ml/               features.py model.py dataset.py train.py train_fast.py gpu_aug.py infer.py
backend/scripts/    fetch_isl_csltr.py import_isl_csltr.py import_word_signs.py import_alphabet.py
                    make_reference_videos.py exp_cv.py exp_dtw.py
frontend/src/       pages/ (Login Intro Setup Studio Live Agent SignBack Spell SignLibrary Collect History Settings Aac)
                    components/ (Shell AvatarStage CameraStage VideoUpload MessageBubble ActionCard Logo)
                    lib/ (api auth prefs speech useHolistic useFaceSignals useSignReader keypoints avatar letters)
data/               sentences.json active_sentences.json signs/ alphabet/ knowledge/ reference_videos/ test_videos/
                    raw/ (own takes, local) external/ letters/ (git-ignored)
models/             isl_transformer.pt cv_report.json eval_report.json (stale)
```

## 13. Git state

- Branch `main`, fully pushed to `origin` as of the last check (head after the retrained model commit `f946d5a`; code commit `a0cbd14`).
  GitHub uploads were flaky at times ("connection reset"); splitting the 14 MB model into its own commit fixed it. A local branch
  `backup-before-split` exists.
- Untracked and intentionally not committed: `models/exp_batch1.sh`, `models/exp_batch3.sh`.
- `.gitignore` also excludes `data/raw/s01,s03,s07,s08`, `models/experiments/`, `models/*_prev.pt`, `datasets/`, `data/external/`, `data/letters/`, `.env`, `*.db`.

## 14. Pitch deck and diagrams (outside the repo)

- Folder `D:\PERSONAL\SRIRAKSHA\PES\Hackathons\AMC-CodeSpark\`. Latest deck: **`Ishara_AI_Pitch_Deck_13slides_v2.pptx`** (built from the
  user's edited `Ishara_AI_Pitch_Deck_11slides.pptx` by `pitch-deck/build/add_diagrams.py`). Order: 1 title, 2 problem, **3 tech stack**, 4 what
  it does, **5 pipeline diagram**, 6 deaf to hearing, 7 recognition, **8 inside the sign reader (Transformer diagram)**, 9 hearing to deaf,
  10 fingerspelling, 11 accessibility agent, 12 own setup, 13 conclusion. Footers and section labels renumbered.
- **Known stale content:** slide 7 still says 54.1% and 99 sentences; the agent and fingerspelling slides predate the live agent and the
  built-in alphabet. Update from the user's latest file into a NEW filename. No em/en dashes.
- Diagrams in `pitch-deck/diagrams/`: `ishara_pipeline` and `ishara_transformer` as `.drawio`, `.svg`, `.png` (black and silver;
  the pipeline includes a two-step "Confidence check" box). Generated by `build/diagrams.py`, rendered through
  `viewer.diagrams.net` in headless Chrome using a small CDP driver (kept in the session scratchpad, not in the repo; re-create if needed:
  encode the .drawio into the viewer URL `#R<deflate+base64>`, scroll the `.geDiagramContainer`, take the SVG, rasterise at 2.5x).
  The user originally asked to draw them in app.diagrams.net via claude-in-chrome; the headless-viewer route was used instead.
- Slide rendering: `pitch-deck/build/render.ps1` (PowerPoint COM) exports PNGs.

## 15. Open items and ideas

- **Docker not rebuilt** since the 20-sentence model, agent rewrite and 25% threshold. Rebuild when asked:
  `docker compose up -d --build` (takes a few minutes; then check http://localhost:8080 serves the new bundle).
- Update the deck's accuracy slide and agent/fingerspelling slides (new file name).
- Real-time speech: optional second confidence check (show the guess briefly and let the signer cancel before it is spoken).
- Agent: optional real Google Calendar write (needs the user's OAuth), send snapshot to the planner (multimodal), more tools.
- More sentences: record 10+ takes each on the Train page (e.g. "What is your name?", "I need help"), add ids to `active_sentences.json`, retrain.
- Camera-dependent features (live recognition, auto-mic, face camera) were only checked in headless Chrome with a fake device or
  not at all with a real camera/microphone. Re-verify by hand.
- Local Qwen translations can be wrong (once translated Friday as Saturday in Kannada). Gemini has no key yet.
- Sandbox browser cannot reach `storage.googleapis.com` (MediaPipe model host); on the user's machine it loads normally.

## 16. How to answer "how did you train it?" (for the user)

"A small Transformer neural network trained from scratch in PyTorch on the public ISL-CSLTR videos plus our own recordings.
MediaPipe turns each video into hand, body and face points; the model matches the movement to a sentence. Trained on a laptop
RTX 5060 and evaluated with 5-fold cross-validation (every video scored by a model that never saw it). 86% top-1 and 96% top-3
on 20 sentences; about 57% on all 99." Disclose that Claude Code was used as a coding assistant if the event asks about AI tools,
and that the chat/translation/agent-planner LLMs (Qwen3.5 local, Gemini backup) were not trained by the team.

## Update: live browser agent, eye calibration, fine-tune UI (2026-11-12)
- **Live browser agent** (`backend/app/services/browser_agent.py`, `routers/browser.py`, `frontend/src/components/BrowserAgent.tsx`): Playwright drives the installed Chrome (headed, persistent profile in `data/browser_profile/`, git-ignored). Each step: read the page as numbered elements, local Qwen picks one JSON action, our code enforces safety. Asks the person for taste choices (identical buttons such as "Add to cart" become a choice with real options unless the goal names it), confirms before Place order / Submit / Request meeting, never types passwords, card numbers, OTPs (asks the person to do it in the window and press Continue). Stable personal details (name, phone, email, address) are remembered per user in `AgentMemory` and auto-filled next time; per-task answers are not stored. Demo sites: `/demo/canteen`, `/demo/portal`. Needs the backend to run on the host (not Docker). Agent page has two tabs: Live browser (default) and Quick plan (older simulated planner).
- **Eye tracking v2** (`lib/gaze.ts`, `components/GazeCalibration.tsx`): features now include MediaPipe eye-direction blendshapes and the 3D head pose from the facial transformation matrix. Guided calibration = 9 dots with settle time and outlier rejection, then a head-turn dot at the centre so head coefficients are learned from data. Leave-one-dot-out error is shown. Every selection nudges a bounded offset; Re-center corrects the offset without discarding the calibration. Quick 3 s start still exists. Only tested on synthetic data so far, not a real face.
- **Train page**: `POST /api/model/train?mode=finetune|full`; `GET /api/model` returns `new_takes` and `last_finetune`. Fine-tune takes about 12 s for 9 new takes (verified with a test run, then reverted).
- Alphabet "loading forever" was a stale dev backend without `/api/alphabet`; the page now shows an error with Try again.
