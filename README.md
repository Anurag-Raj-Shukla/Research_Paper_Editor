# Research Paper Editor (Word Weaver)

> **Status: Ongoing / Work in Progress.** Core features work end-to-end, but this is an active project — some pieces (auth, saving/loading documents, deployment) aren't built yet. See **Roadmap** below.

A browser-based research paper editor with AI-assisted writing tools: a FastAPI backend that runs grammar checking, formality detection, spell checking, and GPT-2 next-word suggestions, paired with a single-file HTML/CSS/JS frontend styled like a document editor.

## Project structure

```
.
├── backend/
│   ├── main.py                 # FastAPI app — all API routes
│   ├── requirements.txt
│   ├── grammar_checker.py      # LanguageTool grammar checking
│   ├── formality_checker.py    # FORMAL / INFORMAL classifier (model + heuristic fallback)
│   ├── spell_checker.py        # pyspellchecker-based spell checking
│   └── word_generator.py       # GPT2 class for next-word suggestions
└── frontend/
    └── index.html               # Whole frontend UI in one file
```

## What each backend module does

- **`grammar_checker.py`** — wraps `language_tool_python` (LanguageTool) to flag grammar issues, skipping pure spelling rules (those are handled separately).
- **`formality_checker.py`** — classifies text as FORMAL or INFORMAL using a HuggingFace model (`s-nlp/roberta-base-formality-ranker`) when available, and falls back to a regex/heuristic scorer (informal contractions, slang, ALL CAPS, sentence length, etc.) if the model can't be downloaded.
- **`spell_checker.py`** — uses `pyspellchecker` for offline spell checking, with a `textblob`-based fallback, and skips a small allowlist of technical abbreviations (`api`, `nlp`, `html`, etc.) so they aren't flagged.
- **`word_generator.py`** — a `GPT2` class (GPT-2 via HuggingFace `transformers`) that generates the next few words after a given text prompt.

## API routes (`backend/main.py`)

| Method | Path              | Purpose                              |
|--------|-------------------|---------------------------------------|
| GET    | `/health`         | Health check                          |
| POST   | `/suggest`        | GPT-2 next-word suggestions           |
| POST   | `/grammar`        | Grammar check via LanguageTool        |
| POST   | `/formality`      | FORMAL / INFORMAL classification      |
| POST   | `/spellcheck/text`| Spell-check the whole document        |
| POST   | `/spellcheck/word`| Spell-check a single word in real time|

## Setup

```bash
cd backend
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

`language_tool_python` downloads and runs a local LanguageTool (Java) server on first use — the first grammar check will be slow while it downloads. Requires a JRE installed on the machine.

## Running it

```bash
cd backend
uvicorn main:app --reload --port 8000
```

Then open `frontend/index.html` directly in a browser (it's hardcoded to talk to `http://localhost:8000`, set via the `API` constant near the top of the `<script>` section — change that if you run the backend on a different host/port).

## Fixes applied in this pass

The backend originally didn't start at all. These have been fixed:

- Removed a dead `from gpt2_suggester import get_suggestions` import — that module never existed in the repo and the function was never used, but the import itself crashed the app on startup.
- Renamed `backend/python/spell_checker` → `backend/spell_checker.py` and `backend/Word_Generation` → `backend/word_generator.py` (both were missing the `.py` extension, so Python couldn't import them).
- Moved `grammar_checker.py`, `formality_checker.py`, and `spell_checker.py` out of `backend/python/` into `backend/` so their flat imports in `main.py` actually resolve.
- Removed leftover Colab script code (`gpt2 = GPT2()` + a `print(...)` call) from the bottom of `word_generator.py` — previously, just *importing* the module would instantiate GPT-2 and run a generation as a side effect.
- Renamed the backend's `/generate` route to `/suggest` so it matches what `frontend/index.html` actually calls (the frontend was already calling `/suggest`; the 404 was on the backend side).
- Added `backend/requirements.txt`, which didn't exist before.

## Roadmap / known gaps

Since this is still ongoing, here's what's intentionally not done yet:

- No persistence — documents aren't saved anywhere; refreshing the page loses your work.
- No authentication/user accounts.
- No automated tests for the backend endpoints.
- CORS is wide open (`allow_origins=["*"]`) — fine for local dev, needs to be locked down before any public deployment.
- Not deployed anywhere yet — currently runs locally only.
- The grammar checker and GPT-2 suggester have no fallback if LanguageTool or the model download fails (the formality/spell checkers do have fallbacks, worth mirroring for the other two).
