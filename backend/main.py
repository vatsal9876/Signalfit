import os
import sys
import threading
from pathlib import Path
from typing import Any, Dict, List, Literal

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field


ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parent
STATIC_DIR = PROJECT_ROOT / "frontend"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

app = FastAPI(
    title="Signalfit SHL Recommendation Agent",
    description="Conversational retrieval API for SHL assessment recommendations.",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

_WARMED = False
_LLM_WARMED = False


class Message(BaseModel):
    model_config = ConfigDict(extra="forbid")

    role: Literal["user", "assistant"]
    content: str = Field(..., min_length=1)


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    messages: List[Message] = Field(..., min_length=1)


class Recommendation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    url: str
    test_type: str


class AgentResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    reply: str
    recommendations: List[Recommendation]
    end_of_conversation: bool


class HealthResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: str


def _to_controller_messages(messages: List[Message]) -> List[Dict[str, str]]:
    return [
        {
            "role": message.role,
            "content": message.content
        }
        for message in messages
    ]


def _run_agent(messages: List[Dict[str, str]]) -> Dict[str, Any]:
    from agent.controller import agent

    try:
        return agent(messages)
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=str(exc)
        ) from exc


def _warm_agent():
    global _WARMED

    if _WARMED:
        return

    from retrieval.embeddings import get_model
    from retrieval.search import catalog, index

    get_model()
    len(catalog)
    index.ntotal
    _WARMED = True


def _warm_llm():
    global _LLM_WARMED

    if _LLM_WARMED or not os.getenv("GROQ_API_KEY"):
        return

    try:
        from groq import Groq

        model = os.getenv(
            "GROQ_STATE_MODEL",
            os.getenv("GROQ_MODEL", "llama-3.1-8b-instant"),
        )
        client = Groq(api_key=os.getenv("GROQ_API_KEY"))

        client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": "Return only valid JSON.",
                },
                {
                    "role": "user",
                    "content": "Warmup. Return {\"ok\": true}.",
                },
            ],
            temperature=0,
            max_tokens=8,
            response_format={"type": "json_object"},
        )
        _LLM_WARMED = True
    except Exception:
        # Warmup should never make the app unavailable.
        _LLM_WARMED = False


@app.on_event("startup")
def startup():
    threading.Thread(target=_warm_llm, daemon=True).start()


@app.get("/")
def root():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health", response_model=HealthResponse)
def health():
    _warm_agent()

    return {
        "status": "ok",
    }


@app.post("/chat", response_model=AgentResponse)
def chat(request: ChatRequest):
    """
    Main conversational endpoint.

    Send the full conversation history. The controller reconstructs current
    state from the full trace, so refinements work without server-side memory.
    """

    messages = _to_controller_messages(request.messages)
    return _run_agent(messages)


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", "8000"))

    uvicorn.run(
        "backend.main:app",
        host="0.0.0.0",
        port=port,
        reload=True
    )
