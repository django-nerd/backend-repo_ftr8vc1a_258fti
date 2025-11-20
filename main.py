import os
from typing import List, Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import requests

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --------- Models ---------
class Message(BaseModel):
    role: str = Field(..., description="system | user | assistant")
    content: str

class ChatRequest(BaseModel):
    messages: List[Message]
    model: Optional[str] = Field(default=None, description="Optional model override")
    temperature: Optional[float] = 0.7

class ChatResponse(BaseModel):
    reply: str
    provider: str
    model: Optional[str] = None


# --------- Routes ---------
@app.get("/")
def read_root():
    return {"message": "Hello from FastAPI Backend!"}

@app.get("/api/hello")
def hello():
    return {"message": "Hello from the backend API!"}

@app.post("/api/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    """Simple chat endpoint. If OPENAI_API_KEY is set, uses OpenAI Chat Completions.
    Otherwise falls back to a lightweight local responder.
    """
    # Basic validation
    if not req.messages or all(not m.content.strip() for m in req.messages):
        raise HTTPException(status_code=400, detail="No messages provided")

    api_key = os.getenv("OPENAI_API_KEY")
    model = req.model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    # Try OpenAI if key available
    if api_key:
        try:
            payload = {
                "model": model,
                "temperature": req.temperature,
                "messages": [m.model_dump() for m in req.messages],
            }
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            }
            resp = requests.post(
                "https://api.openai.com/v1/chat/completions",
                json=payload,
                headers=headers,
                timeout=30,
            )
            if resp.status_code != 200:
                # Fallback gracefully if API fails
                raise RuntimeError(f"OpenAI error: {resp.status_code} {resp.text[:200]}")
            data = resp.json()
            # Extract assistant reply
            reply = (
                data.get("choices", [{}])[0]
                .get("message", {})
                .get("content", "")
                .strip()
            )
            if not reply:
                reply = "I'm here, but I couldn't generate a response just now."
            return ChatResponse(reply=reply, provider="openai", model=model)
        except Exception as e:
            # Fall through to local responder
            fallback = local_responder(req.messages)
            return ChatResponse(reply=f"(fallback) {fallback}", provider="local", model=None)

    # No API key – use local responder
    reply = local_responder(req.messages)
    return ChatResponse(reply=reply, provider="local", model=None)


def local_responder(messages: List[Message]) -> str:
    """A tiny, deterministic helper to simulate an assistant when no API key is set.
    It:
    - Echoes the last user message with a friendly preface
    - Adds one or two tips for better results
    """
    last_user = next((m.content.strip() for m in reversed(messages) if m.role == "user" and m.content.strip()), "")
    if not last_user:
        return "Tell me what you'd like to create, and I'll help step-by-step."

    tips = (
        "I don't have external AI access in this environment, so this is a simulated reply. "
        "You can add your OpenAI API key to enable real AI responses. "
        "Meanwhile, here's how I'd start:"
    )
    return (
        f"You said: ‘{last_user}’.\n\n"
        f"{tips}\n\n"
        f"- Break the request into clear steps\n"
        f"- Share any examples or constraints\n"
        f"- I can generate UI and backend code for you here"
    )


@app.get("/test")
def test_database():
    """Test endpoint to check if database is available and accessible"""
    response = {
        "backend": "✅ Running",
        "database": "❌ Not Available",
        "database_url": None,
        "database_name": None,
        "connection_status": "Not Connected",
        "collections": []
    }
    
    try:
        # Try to import database module
        from database import db
        
        if db is not None:
            response["database"] = "✅ Available"
            response["database_url"] = "✅ Configured"
            response["database_name"] = db.name if hasattr(db, 'name') else "✅ Connected"
            response["connection_status"] = "Connected"
            
            # Try to list collections to verify connectivity
            try:
                collections = db.list_collection_names()
                response["collections"] = collections[:10]  # Show first 10 collections
                response["database"] = "✅ Connected & Working"
            except Exception as e:
                response["database"] = f"⚠️  Connected but Error: {str(e)[:50]}"
        else:
            response["database"] = "⚠️  Available but not initialized"
            
    except ImportError:
        response["database"] = "❌ Database module not found (run enable-database first)"
    except Exception as e:
        response["database"] = f"❌ Error: {str(e)[:50]}"
    
    # Check environment variables
    import os
    response["database_url"] = "✅ Set" if os.getenv("DATABASE_URL") else "❌ Not Set"
    response["database_name"] = "✅ Set" if os.getenv("DATABASE_NAME") else "❌ Not Set"
    
    return response


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
