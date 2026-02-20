import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from pymongo import MongoClient
from datetime import datetime
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel #maintain structure of the question(input) and answer

load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")
mongo_uri=os.getenv("MONGO_URI")
client=MongoClient(mongo_uri)
db=client["chat"]
collection=db["users"]

app=FastAPI()

class ChatRequest(BaseModel):   
    user_id: str
    question: str

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],#all devices/systems can access the api
    allow_methods=["*"], 
    allow_headers=["*"],
    allow_credentials=True
)

#to store memory (getting or finding name of the user)
def get_history(user_id):
    chats=collection.find({"user_id": user_id}).sort("timestamp",1) #based on userid it will find all messages
    history=[] #empty list to store the history of chats

    for chat in chats:
        history.append((chat["role"],chat["message"])) #appending role and message to the list
    return history

prompt=ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful fitness assistant."), #based on this output changes
        ("placeholder", "{history}"),#passing the history of chats to the prompt template
        ("user", "{question}")
    ]
)

llm=ChatGroq(api_key=groq_api_key,model="openai/gpt-oss-20b")
chain= prompt | llm

@app.get("/") #root route (get->getting info)
#if someone hits the root route it will return a message for that the below func
def home():
    return {"message":"Welcome to the fitness assistant chatbot API!"}

@app.post("/chat") #post->posting info/share some info
def chat(request: ChatRequest):
    history=get_history(request.user_id) #getting the history of chats based on user id
    response = chain.invoke({"history": history, "question": request.question})#before taking question taking the history of that user 
    #storing in db
    collection.insert_one({"user_id": request.user_id, 
                       "role": "user",
                       "message": request.question,
                       "timestamp":datetime.utcnow()})   #user query stored in mongodb
    collection.insert_one({"user_id": request.user_id, 
                       "role": "assistant",
                       "message": response.content,
                       "timestamp":datetime.utcnow()})   #assistant response stored in mongodb  

    return {"response": response.content}
    
