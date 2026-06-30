# SHL Conversational Assessment Recommender

A stateless FastAPI service that acts as a conversational AI agent for recommending SHL assessment products. Built for the SHL hiring assessment.

## Architecture

- **Stateless**: Every `/chat` call carries the full conversation history. No server-side session storage.
- **Async**: FastAPI with async handlers + AsyncGroq client for concurrent request handling.
- **Retrieval-grounded**: TF-IDF search over the real SHL catalog. The LLM interprets intent; the retrieval layer is the sole source of truth for recommendations.

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Set up environment
```bash
cp .env.example .env
# Edit .env and add your GROQ_API_KEY
```

### 3. Download catalog (one-time)
```bash
python -m scripts.download_catalog
```

### 4. Run the server
```bash
uvicorn app.main:app --reload --port 8000
```

### 5. Test
```bash
# Health check
curl http://localhost:8000/health

# Chat
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "I need a Python programming test"}]}'
```

## Endpoints

### `GET /health`
Returns `{"status": "ok"}` with HTTP 200.

### `POST /chat`
**Request:**
```json
{
  "messages": [
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "..."}
  ]
}
```

**Response:**
```json
{
  "reply": "string",
  "recommendations": [
    {"name": "string", "url": "string", "test_type": "string"}
  ],
  "end_of_conversation": false
}
```

## Testing

```bash
# Schema compliance tests
pytest tests/test_schema.py -v

# Behavior probe tests
pytest tests/test_behaviors.py -v

# Evaluation harness (requires running server + trace files)
python -m tests.eval_harness --base-url http://localhost:8000 --traces-dir traces/
```

## Project Structure

```
├── app/
│   ├── main.py          # FastAPI app, endpoints, CORS, startup
│   ├── models.py        # Pydantic request/response models
│   ├── catalog.py       # Catalog loading and lookup
│   ├── retrieval.py     # TF-IDF search over catalog
│   ├── llm_client.py    # Async Groq client with timeout/retry
│   ├── agent.py         # Core agent logic (intent routing)
│   └── prompts.py       # System prompts for LLM calls
├── data/
│   └── catalog.json     # Pre-processed catalog (370 items)
├── scripts/
│   └── download_catalog.py
├── tests/
│   ├── test_schema.py
│   ├── test_behaviors.py
│   └── eval_harness.py
├── traces/              # Place 10 public trace .md files here
├── requirements.txt
├── Procfile
└── .env.example
```

## Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Retrieval | TF-IDF + keyword | Small catalog (370 items), fast startup, no model download |
| LLM | Groq llama-3.3-70b-versatile | Fast inference, strong instruction following |
| Schema enforcement | Pydantic + validation guard | LLM never touches wire format directly |
| Job Solutions filter | Exclude "Solution" in name | 7 items excluded; conservative name heuristic |
| test_type | Full join of keys → letter codes | Dominant pattern in traces |
