import pandas as pd
import tez
import torch
import torch.nn as nn
import transformers

from typing import Optional
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class SentimentPredict(BaseModel):
    lyrics: str
    #title: str
    threshold: float


class BERTDataset:
    def __init__(self, lyrics, target):
        self.lyrics = lyrics
        self.target = target
        self.tokenizer = transformers.BertTokenizer.from_pretrained(
            "sagorsarker/bangla-bert-base", do_lower_case=True
        )
        self.max_len = 500

    def __len__(self):
        return len(self.lyrics)

    def __getitem__(self, item):
        lyrics = str(self.lyrics[item])
        lyrics = " ".join(lyrics.split())

        inputs = self.tokenizer.encode_plus(
            lyrics,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
        )

        ids = inputs["input_ids"]
        mask = inputs["attention_mask"]
        token_type_ids = inputs["token_type_ids"]

        return {
            "ids": torch.tensor(ids, dtype=torch.long),
            "mask": torch.tensor(mask, dtype=torch.long),
            "token_type_ids": torch.tensor(token_type_ids, dtype=torch.long),
            "targets": torch.tensor(self.target[item], dtype=torch.float),
        }



class BERTBaseUncased(tez.Model):
    def __init__(self):
        super().__init__()
        self.tokenizer = transformers.BertTokenizer.from_pretrained(
            "sagorsarker/bangla-bert-base", do_lower_case=True
        )
        self.bert = transformers.BertModel.from_pretrained("sagorsarker/bangla-bert-base")
        self.bert_drop = nn.Dropout(0.3)
        self.out = nn.Linear(768,1)


    def forward(self, ids, mask, token_type_ids, targets=None):
        _, o_2 = self.bert(ids, attention_mask=mask, token_type_ids=token_type_ids)
        b_o = self.bert_drop(o_2)
        output = torch.sigmoid(self.out(b_o))
        return output, 0, {}

MODEL = BERTBaseUncased()
MODEL.load("G:\MusicMood\MusicModel.bin", device="cpu")        

        
@app.get("/")
def read_root():
    return {"Hello": "World"}
        

@app.post("/predict")
def fetch_predections(sp : SentimentPredict):
    dataset = BERTDataset([sp.lyrics],[-1])
    prediction = float(list(MODEL.predict(dataset, batch_size=1))[0][0][0])
    Mood= "আনন্দ" 
    if prediction > sp.threshold:
        Mood = "বেদনা"
    return {"বেদনা":prediction, "আনন্দ": 1-prediction, "lyrics": sp.lyrics, "Mood": Mood}

@app.get("/predict")
def fetch_predection(lyrics: str):
    dataset = BERTDataset([lyrics],[-1])
    prediction = float(list(MODEL.predict(dataset, batch_size=1))[0][0][0])      
    return {"বেদনা":prediction, "আনন্দ": 1-prediction, "lyrics": lyrics}  