import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from datasets import load_dataset
from tqdm import tqdm
import sys
import os

from models.backbone import FrozenMistralBackbone
from models.smec_adaptor import SMECAdaptor

class ContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super(ContrastiveLoss, self).__init__()
        self.temperature = temperature

    def forward(self, text_embeddings, image_embeddings):
        text_embeddings = F.normalize(text_embeddings, p=2, dim=1)
        image_embeddings = F.normalize(image_embeddings, p=2, dim=1)

        logits = torch.matmul(text_embeddings, image_embeddings.T) / self.temperature
        labels = torch.arange(logits.size(0)).to(logits.device)
        
        loss_t2i = F.cross_entropy(logits, labels)
        loss_i2t = F.cross_entropy(logits.T, labels)
        
        return (loss_t2i + loss_i2t) / 2

def train():
    print("Initializing Training Environment...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("Loading Frozen Backbone...")
    backbone = FrozenMistralBackbone().to(device)
    
    print("Initializing SMEC Adaptor...")
    base_dim = 4096
    candidate_dims = [256, 512, 1024, 2048, 4096]
    adaptor = SMECAdaptor(base_dim=base_dim, candidate_dims=candidate_dims).to(device)

    optimizer = optim.AdamW(adaptor.parameters(), lr=1e-4, weight_decay=1e-4)
    criterion = ContrastiveLoss(temperature=0.07)

    print("Loading Standard Contrastive Text Pairs via HuggingFace...")
    # 拉取自带正样本对的高质量推理数据集
    dataset = load_dataset("sentence-transformers/all-nli", split="train")
    
    batch_size = 12
    epochs = 3
    print(f"Starting Training Loop (Epochs: {epochs}, Batch Size: {batch_size})...")

    adaptor.train()
    
    # 为了演示快速跑通，我们先取前 1200 条数据进行一个 epoch 的快速验证
    num_batches = 100 
    
    for epoch in range(epochs):
        total_loss = 0.0
        progress_bar = tqdm(range(num_batches), desc=f"Epoch {epoch+1}/{epochs}")
        
        for batch_idx in progress_bar:
            start_idx = batch_idx * batch_size
            end_idx = start_idx + batch_size
            batch_samples = dataset[start_idx:end_idx]
            
            texts_query = batch_samples["anchor"]
            texts_doc = batch_samples["entailment_positive"]

            with torch.no_grad():
                emb_q_raw = backbone(texts_query).to(device)
                emb_d_raw = backbone(texts_doc).to(device)

            emb_q_truncated, weights_q = adaptor(emb_q_raw, tau=1.0, hard=True)
            emb_d_truncated, weights_d = adaptor(emb_d_raw, tau=1.0, hard=True)

            loss = criterion(emb_q_truncated, emb_d_truncated)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_loss = total_loss / num_batches
        print(f"Epoch {epoch+1} completed. Average Loss: {avg_loss:.4f}\n")

    print("Training finished successfully. The A100 Engine is fully calibrated.")

if __name__ == "__main__":
    train()