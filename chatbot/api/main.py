from fastapi import FastAPI
from pydantic import BaseModel

import sys
import os
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(1, '/home/inamurahman/LibraryManagement-AI')
from chatbot.agent.main_graph import chat_with_agent

app = FastAPI()

class ChatUser(BaseModel):
    user_message: str
    auth_token: str
    image_base64: str = None


@app.get("/")
async def read_root():
    return {"message": "Welcome to the Library Assistant API!"}


@app.post("/chat")
async def chat_endpoint(chat_user: ChatUser):
    response = chat_with_agent(chat_user.user_message, chat_user.auth_token, chat_user.image_base64)
    print("Response from chat_with_agent:", response)
    return {"output": response}

