# import sys
# import os

# sys.path.append(os.path.abspath(".."))
# from package import *

# from fastapi import FastAPI
# from fastapi.middleware.cors import CORSMiddleware
# from pydantic import BaseModel


#app = FastAPI()

from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def root():
    return {"message": "FastAPI running on EC2 🚀"}

# app.add_middleware(
#     CORSMiddleware,
#     #allow_origins=["http://localhost:3000","http://10.186.103.211:3000"],  # your Next.js app
#     allow_origins=["*"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# @app.get("/")
# def home():
#     return {"message": "API is working"}
