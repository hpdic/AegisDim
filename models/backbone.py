import torch
import torch.nn as nn
from transformers import AutoTokenizer
from llm2vec import LLM2Vec
import os
import sys

sys.stderr = open(os.devnull, 'w')

class FrozenMistralBackbone(nn.Module):
    def __init__(self, model_name="mistralai/Mistral-7B-v0.1",
                 peft_path="McGill-NLP/LLM2Vec-Mistral-7B-Instruct-v2-mntp"):
        super().__init__()
        print("Initializing Tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        print("Loading and freezing Mistral 7B backbone via LLM2Vec...")
        self.l2v = LLM2Vec.from_pretrained(
            base_model_name_or_path=model_name,
            peft_model_name_or_path=peft_path,
            device_map="auto",
            torch_dtype=torch.bfloat16
        )
        for param in self.l2v.parameters():
            param.requires_grad = False
        self.l2v.eval()
        print("Backbone loaded and frozen.")

    def forward(self, texts):
        self.l2v.eval()
        with torch.no_grad():
            embeddings = self.l2v.encode(texts)
        # 转换为 float32 便于后续相似度计算
        return embeddings.float()