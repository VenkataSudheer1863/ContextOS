"""FastAPI application for ContextOS context management service."""
from __future__ import annotations
import time
from typing import Any, Dict, List, Optional
from fastapi import FastAPI, HTTPException, WebSocket, Depends, Security
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field
import uvicorn
from loguru import logger

# ---- Pydantic Models ----
class QueryRequest(BaseModel):
    query: str
    agent_id: str
    max_tokens: int = 8192
    task_type: str = "general"
    session_id: str = ""
    include_working_memory: bool = True

class ContextItemResponse(BaseModel):
    id: str
    content: str
    memory_type: str
    importance: float
    relevance: float
    token_count: int

class ContextResponse(BaseModel):
    items: List[ContextItemResponse]
    total_tokens: int
    compressed: bool
    context_string: str
    processing_time_ms: float
    stats: Dict[str, Any]

class MemoryStoreRequest(BaseModel):
    content: str
    memory_type: str = "observation"
    importance: float = Field(default=0.5, ge=0.0, le=1.0)
    agent_id: str
    metadata: Dict[str, Any] = {}

class AgentRunRequest(BaseModel):
    task: str
    agent_id: str
    max_iterations: int = 10
    context_config: Dict[str, Any] = {}

class StatsResponse(BaseModel):
    total_tokens_used: int
    total_items_retrieved: int
    total_items_compressed: int
    governance_evictions: int
    scheduling_overhead_ms: float
    uptime_seconds: float

# ---- App Setup ----
app = FastAPI(
    title="ContextOS API",
    description="Context lifecycle management for autonomous agents",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_orchestrators: Dict[str, Any] = {}
_start_time = time.time()
API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)

def get_orchestrator(agent_id: str):
    if agent_id not in _orchestrators:
        from ..core.orchestrator import ContextOrchestrator, AgentState
        _orchestrators[agent_id] = ContextOrchestrator()
    return _orchestrators[agent_id]

@app.get("/health")
async def health():
    return {"status": "ok", "uptime": time.time() - _start_time}

@app.post("/context/process", response_model=ContextResponse)
async def process_context(req: QueryRequest):
    t0 = time.perf_counter()
    try:
        from ..core.orchestrator import AgentState
        orch = get_orchestrator(req.agent_id)
        state = AgentState(agent_id=req.agent_id, current_task=req.query,
                          task_type=req.task_type, session_id=req.session_id)
        result = orch.process_query(req.query, state)
        elapsed = (time.perf_counter() - t0) * 1000
        return ContextResponse(
            items=[ContextItemResponse(id=i.id, content=i.content,
                    memory_type=i.memory_type.value, importance=i.importance,
                    relevance=i.relevance, token_count=i.token_count)
                   for i in result.items],
            total_tokens=result.total_tokens,
            compressed=result.compressed,
            context_string=result.context_string,
            processing_time_ms=elapsed,
            stats={"items_retrieved": result.stats.total_items_retrieved,
                   "tokens_used": result.stats.total_tokens_used},
        )
    except Exception as e:
        logger.exception(e)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/memory/store")
async def store_memory(req: MemoryStoreRequest):
    try:
        from ..core.orchestrator import MemoryType
        orch = get_orchestrator(req.agent_id)
        mtype = MemoryType(req.memory_type)
        item_id = orch.store_observation(req.content, mtype, req.importance, req.metadata)
        return {"id": item_id, "status": "stored"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/memory/working/{agent_id}")
async def get_working_memory(agent_id: str):
    orch = get_orchestrator(agent_id)
    if hasattr(orch, "working_memory"):
        items = orch.working_memory.get_all()
        return {"items": [{"id": i.id, "content": i.content[:100], "type": i.memory_type.value} for i in items],
                "count": len(items)}
    return {"items": [], "count": 0}

@app.post("/memory/governance/{agent_id}")
async def run_governance(agent_id: str):
    orch = get_orchestrator(agent_id)
    evicted = orch.run_governance_cycle()
    return {"evicted_count": len(evicted), "evicted_ids": evicted}

@app.get("/stats")
async def get_stats():
    total_tokens = sum(getattr(o, "_stats", {}).total_tokens_used if hasattr(getattr(o, "_stats", None), "total_tokens_used") else 0
                       for o in _orchestrators.values())
    return StatsResponse(total_tokens_used=total_tokens, total_items_retrieved=0,
                         total_items_compressed=0, governance_evictions=0,
                         scheduling_overhead_ms=0.0, uptime_seconds=time.time() - _start_time)

@app.websocket("/agents/stream/{agent_id}")
async def agent_stream(websocket: WebSocket, agent_id: str):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_json()
            query = data.get("query", "")
            from ..core.orchestrator import AgentState
            orch = get_orchestrator(agent_id)
            state = AgentState(agent_id=agent_id, current_task=query)
            result = orch.process_query(query, state)
            await websocket.send_json({"context": result.context_string[:500], "tokens": result.total_tokens})
    except Exception:
        await websocket.close()

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
