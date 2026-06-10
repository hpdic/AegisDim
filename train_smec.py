import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import sys
import os

from models.backbone import FrozenMistralBackbone
from models.smec_adaptor import SMECAdaptor

# --- 多模态对比损失函数 ---
class ContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super(ContrastiveLoss, self).__init__()
        self.temperature = temperature

    def forward(self, text_embeddings, image_embeddings):
        # 归一化特征
        text_embeddings = F.normalize(text_embeddings, p=2, dim=1)
        image_embeddings = F.normalize(image_embeddings, p=2, dim=1)

        # 计算相似度矩阵
        logits = torch.matmul(text_embeddings, image_embeddings.T) / self.temperature
        
        # 目标是让对角线上的正样本相似度最高
        labels = torch.arange(logits.size(0)).to(logits.device)
        
        # 对称的交叉熵损失
        loss_t2i = F.cross_entropy(logits, labels)
        loss_i2t = F.cross_entropy(logits.T, labels)
        
        return (loss_t2i + loss_i2t) / 2

# --- 主训练循环 ---
def train():
    print("Initializing Training Environment...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. 加载冻结的主干网络 (Mistral 7B)
    print("Loading Frozen Backbone...")
    backbone = FrozenMistralBackbone().to(device)
    
    # 2. 初始化 SMEC 适配器
    print("Initializing SMEC Adaptor...")
    base_dim = 4096
    candidate_dims = [256, 512, 1024, 2048, 4096]
    adaptor = SMECAdaptor(base_dim=base_dim, candidate_dims=candidate_dims).to(device)

    # 3. 设置优化器和损失函数
    # 注意：我们只训练 adaptor 的参数！
    optimizer = optim.AdamW(adaptor.parameters(), lr=1e-4, weight_decay=1e-4)
    criterion = ContrastiveLoss(temperature=0.07)

    # TODO: 4. 加载真实的数据集 (Dataloader)
    # 这里为了演示主循环跑通，我们先构造一些假数据
    # 之后我们会把它换成真正的文本-图像对，或者正负样本文本对
    batch_size = 16
    epochs = 3
    print(f"Starting Training Loop (Epochs: {epochs}, Batch Size: {batch_size})...")

    adaptor.train()
    
    for epoch in range(epochs):
        total_loss = 0.0
        # 这里用一个简单的循环模拟 dataloader 的行为
        num_batches = 10 
        progress_bar = tqdm(range(num_batches), desc=f"Epoch {epoch+1}/{epochs}")
        
        for batch_idx in progress_bar:
            # --- 模拟数据加载 ---
            # 在真实的跨模态任务中，这里会是成对的文本和图像（或者文本和正例/负例文本）
            # 我们先用两个不同的文本 batch 模拟两个模态的输入
            texts_a = [f"Text query {i} for batch {batch_idx}" for i in range(batch_size)]
            texts_b = [f"Positive document {i} for batch {batch_idx}" for i in range(batch_size)]

            # --- 前向传播：提取特征 ---
            # 这一步是不计算梯度的
            with torch.no_grad():
                emb_a_raw = backbone(texts_a).to(device)
                emb_b_raw = backbone(texts_b).to(device)

            # --- 自适应降维 ---
            # 数据流经我们的外挂适配器，这里开始有梯度流动
            emb_a_truncated, weights_a = adaptor(emb_a_raw, tau=1.0, hard=True)
            emb_b_truncated, weights_b = adaptor(emb_b_raw, tau=1.0, hard=True)

            # --- 计算对比损失 ---
            loss = criterion(emb_a_truncated, emb_b_truncated)

            # --- 反向传播与优化 ---
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_loss = total_loss / num_batches
        print(f"Epoch {epoch+1} completed. Average Loss: {avg_loss:.4f}\n")

    print("Training finished successfully.")

if __name__ == "__main__":
    train()