from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
import httpx
import os

app = FastAPI()

# Allow all CORS for MVP simplicity
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

HUGGINGFACE_API_URL = "https://api-inference.huggingface.co/models/facebook/blenderbot-400M-distill"
HUGGINGFACE_API_TOKEN = os.getenv("HF_API_TOKEN")  # Set this in your deployment environment for higher rate limits

@app.post("/chat")
async def chat(request: Request):
    data = await request.json()
    user_message = data.get("message", "")
    if not user_message:
        return {"error": "No message provided."}

    headers = {"Authorization": f"Bearer {HUGGINGFACE_API_TOKEN}"} if HUGGINGFACE_API_TOKEN else {}
    payload = {"inputs": user_message}

    async with httpx.AsyncClient() as client:
        response = await client.post(HUGGINGFACE_API_URL, json=payload, headers=headers)
        if response.status_code == 200:
            result = response.json()
            # Hugging Face returns a list of generated texts
            if isinstance(result, dict) and "generated_text" in result:
                return {"response": result["generated_text"]}
            elif isinstance(result, list) and len(result) > 0 and "generated_text" in result[0]:
                return {"response": result[0]["generated_text"]}
            else:
                return {"response": str(result)}
        else:
            return {"error": f"Hugging Face API error: {response.text}"}

@app.get("/")
def root():
    return {"message": "AI Chatbot backend is running!"} 