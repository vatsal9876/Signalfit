# Signalfit

conversational retrieval for recruitment assessment recommendation

## FastAPI app

Run locally:

```bash
uvicorn backend.main:app --reload
```

Production/deployment command:

```bash
uvicorn backend.main:app --host 0.0.0.0 --port $PORT
```

Endpoints:

- `GET /health` - service health and Groq configuration check
- `POST /recommend` - one-shot recommendation request
- `POST /chat` - full conversation request for clarification/refinement traces

Example `/recommend` request:

```json
{
  "query": "Hiring a mid-level Java backend engineer with stakeholder communication"
}
```

Example `/chat` request:

```json
{
  "messages": [
    {
      "role": "user",
      "content": "We need a solution for senior leadership."
    },
    {
      "role": "assistant",
      "content": "Is this assessment for selection/hiring, or for leadership development?"
    },
    {
      "role": "user",
      "content": "For leadership development."
    }
  ]
}
```

The app is stateless: send full conversation history for refinement and comparison turns.

shl_agent/
│
├── backend/
│   ├── main.py
│   ├── requirements.txt
│   │
│   ├── data/
│   │   ├── shl_catalog.json
│   │   ├── embeddings.npy
│   │   └── data_index
│   │
│   ├── retrieval/
│   │   ├── embeddings.py
│   │   ├── indexer.py
│   │   └── search.py
│   │
│   ├── agent/
│   │   ├── controller.py
│   │   ├── operations.py
│   │   ├── state.py
│   │   ├── reranker.py
│   │   └── prompts.py
│   │
│   └── evaluation/
│       └── eval.py
│
└── frontend/
    ├── index.html
    ├── app.js
    └── styles.css
