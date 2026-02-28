# import sys
# import os

# sys.path.append(os.path.abspath(".."))
# from package import *

# from fastapi import FastAPI
# from fastapi.middleware.cors import CORSMiddleware
# from pydantic import BaseModel


#app = FastAPI()

from package import *
import torch

from fastapi import FastAPI

app = FastAPI()

device = torch.device("cpu")
tokenizer = SingleCharTokenizer()
tokens = torch.tensor(tokenizer.load_tokens("tokens_sc.tok")).to(device)

init = "bonjour, je m'appelle chateaubriand. je suis un homme de mon temps, j'ai connu les voyages, la revolutions et les cataclysmes politiques et je pense que la revolution a eu un impact catastrophique sur la france. pour l'affirmer je fais appel a l'autorite du pape leon. bonjour, je m'appelle chateaubriand. je suis un homme de mon temps, j'ai connu les voyages, la revolutions et les cataclysmes politiques et je pense que la revolution a eu un impact catastrophique sur la france. pour l'affirmer je fais appel a l'autorite du pape leon. bonjour, je m'appelle chateaubriand. je suis un homme de mon temps, j'ai connu les voyages, la revolutions et les cataclysmes politiques. "

init_tok = tokenizer.encode(init)

model = torch.load("gpt_cpu.w").to(device)
model.eval()




print('API starting...')



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

