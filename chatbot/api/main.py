from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import sys
import os
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(1, '/home/inamurahman/library-management-system/LibraryManagement-AI')
from chatbot.agent.main_graph import chat_with_agent, stream_chat_with_agent

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods (GET, POST, etc.)
    allow_headers=["*"],  # Allows all headers
)

class ChatUser(BaseModel):
    message: str
    auth_token: str
    image_base64: str = None


@app.get("/")
async def read_root():
    return {"message": "Welcome to the Library Assistant API!"}


@app.post("/chat")
async def chat_endpoint(chat_user: ChatUser):
    response = chat_with_agent(chat_user.message, chat_user.auth_token, chat_user.image_base64)
    print("Response from chat_with_agent:", response)

    return {"message": response.get("message", ""), "books": response.get("books", []).books if response.get("books") else []}

@app.websocket("/chat/{user_id}")
async def websocket_endpoint(websocket: WebSocket, user_id: str):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_text()
            print(f"Received message from user {user_id}: {data}")
            async for event in stream_chat_with_agent(data, user_id, image_base64=None):
                print(f"Sending event to user {user_id}: {event}")
                await websocket.send("hi")

    except Exception as e:
        print(f"Error in websocket for user {user_id}: {e}")
    finally:
        await websocket.close()
