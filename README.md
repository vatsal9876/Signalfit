# Signalfit

conversational retrieval for recruitment assessment recommendation

shl_agent/
│
├── app.py
├── requirements.txt
│
├── data/
│   ├── shl_catalog.json
│   └── shl.index
│
├── retrieval/
│   ├── embeddings.py
│   ├── indexer.py
│   └── search.py
│
├── agent/
│   ├── controller.py
│   ├── operations.py
│   ├── state.py
│   └── prompts.py
│
├── utils/
│   └── helpers.py
│
└── evaluation/
    └── eval.py