from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from fastapi.middleware.cors import CORSMiddleware
from llmHelper import getLlmResponse
# from data import emails

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  
    allow_credentials=True,
    allow_methods=["*"],  
    allow_headers=["*"],  
)

class QueryRequest(BaseModel):
    query: str

@app.post("/submit-query/")
async def submit_query(request: QueryRequest):
    response = getLlmResponse(request.query)
    
    return {
        "message": "Query received successfully",
        "question": request.query,
        "llm_response": response  
    }