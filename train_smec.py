import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from datasets import load_dataset
from collections import deque
import os
import sys

# 屏蔽 stderr 进度条
sys.stderr = open(os.devnull, 'w')

from models.backbone import FrozenMistralBackbone
from models.smec_adaptor import SMECAdaptor


# ---------- Rank Loss (Eq. 8) ----------
def rank_loss(emb_q, emb_d, dim=None):
    """In-batch rank loss. emb_q and emb_d are (B, D).
       Uses in-batch negatives for each anchor-positive pair."""
    B = emb_q.size(0)
    if dim is not None:
        emb_q = emb_q[:, :dim]
        emb_d = emb_d[:, :dim]

    sim = torch.matmul(F.normalize(emb_q, dim=1),
                       F.normalize(emb_d, dim=1).t())  # (B, B)
    pos_sim = sim.diag().unsqueeze(1)                  # (B, 1)
    neg_sim = sim * (1 - torch.eye(B, device=sim.device))  # zero diag

    # For each positive, all other docs are negatives (I(y_ij>y_ik) always true)
    diff = neg_sim - pos_sim
    loss = torch.log(1 + torch.exp(diff))   # (B, B)
    loss = loss.sum() / (B * (B - 1))
    return loss


# ---------- S-XBM Module ----------
class SXBM:
    def __init__(self, memory_size=5000, topk=10):
        self.memory_size = memory_size
        self.topk = topk
        self.queue = deque(maxlen=memory_size)
        self.queue_tensor = None

    def enqueue(self, features):
        """features: (B, D) detached cpu tensor"""
        for f in features.cpu():
            self.queue.append(f)
        self.update_tensor()

    def update_tensor(self):
        if len(self.queue) > 0:
            self.queue_tensor = torch.stack(list(self.queue), dim=0)  # (M, D)
        else:
            self.queue_tensor = None

    def get_topk(self, queries, k=None):
        """queries: (B, D) on GPU. Returns topk indices (B, K) in memory."""
        k = k or self.topk
        if self.queue_tensor is None or self.queue_tensor.size(0) < k:
            return None
        mem = self.queue_tensor.to(queries.device)
        queries_n = F.normalize(queries, dim=1)
        mem_n = F.normalize(mem, dim=1)
        sim = torch.matmul(queries_n, mem_n.t())  # (B, M)
        _, idx = torch.topk(sim, k=k, dim=1)
        return idx

    def compute_unsup_loss(self, query_high, query_low, adaptor, stage_idx):
        """
        Eq. 7: Σ_i,j |sim(high_i, high_j) - sim(low_i, low_j)|
        query_high: (B, D) original high-dim embeddings of current batch
        query_low : (B, d) low-dim embeddings after current stage
        adaptor   : SMECAdaptor (to compute low-dim for memory items)
        stage_idx : current stage index (0-based)
        """
        if self.queue_tensor is None:
            return torch.tensor(0.0, device=query_high.device)

        topk_idx = self.get_topk(query_high, k=self.topk)  # (B, K)
        if topk_idx is None:
            return torch.tensor(0.0, device=query_high.device)

        B, K = topk_idx.shape
        mem_high = self.queue_tensor.to(query_high.device)[topk_idx]  # (B, K, D)

        # Compute low-dim embeddings of memory items using the adaptor up to current stage
        mem_high_flat = mem_high.reshape(B * K, -1)  # (B*K, D)
        with torch.no_grad():
            # Run all previous frozen stages
            mem_low = mem_high_flat
            for i in range(stage_idx):
                mem_low, _ = adaptor.stages[i](mem_low)
            # Run current stage (trainable) – no gradient on memory
            mem_low, _ = adaptor.stages[stage_idx](mem_low)
        mem_low = mem_low.view(B, K, -1)  # (B, K, d)

        # Compute high-dim similarities
        q_high_n = F.normalize(query_high, dim=1).unsqueeze(1)       # (B, 1, D)
        m_high_n = F.normalize(mem_high, dim=2)                      # (B, K, D)
        sim_high = (q_high_n * m_high_n).sum(dim=2)                  # (B, K)

        # Compute low-dim similarities
        q_low_n = F.normalize(query_low, dim=1).unsqueeze(1)         # (B, 1, d)
        m_low_n = F.normalize(mem_low, dim=2)                        # (B, K, d)
        sim_low = (q_low_n * m_low_n).sum(dim=2)                     # (B, K)

        loss = F.l1_loss(sim_low, sim_high.detach())                 # Eq. 7
        return loss


# ---------- Training Loop ----------
def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Backbone
    backbone = FrozenMistralBackbone().to(device)

    # 2. Adaptor stages: 4096 -> 2048 -> 1024 -> 512 -> 256
    stage_dims = [4096, 2048, 1024, 512, 256]
    adaptor = SMECAdaptor(stage_dims).to(device)

    # 3. Dataset
    dataset = load_dataset("sentence-transformers/all-nli", "pair", split="train")

    # 4. S-XBM
    sxbm = SXBM(memory_size=5000, topk=10)

    batch_size = 16
    epochs_per_stage = 10
    num_batches = 200

    print("=== SMEC Sequential Training ===")

    for stage_idx, stage in enumerate(adaptor.stages):
        print(f"\n--- Stage {stage_idx+1}: {stage.input_dim} -> {stage.output_dim} ---")

        # Freeze previous stages
        for i in range(stage_idx):
            adaptor.stages[i].eval()
            for p in adaptor.stages[i].parameters():
                p.requires_grad = False
        # Unfreeze current stage
        stage.train()
        for p in stage.parameters():
            p.requires_grad = True

        optimizer = optim.AdamW(stage.parameters(), lr=1e-4, weight_decay=1e-4)

        for epoch in range(epochs_per_stage):
            total_rank = 0.0
            total_unsup = 0.0

            for batch_idx in range(num_batches):
                start = (batch_idx * batch_size) % len(dataset)
                batch = dataset[start:start + batch_size]
                texts_q = batch["anchor"]
                texts_d = batch["positive"]

                # 1. High-dim embeddings from frozen backbone
                with torch.no_grad():
                    emb_q_high = backbone(texts_q).to(device)
                    emb_d_high = backbone(texts_d).to(device)

                # 2. Input to current stage (run previous frozen stages)
                emb_q_in = emb_q_high
                emb_d_in = emb_d_high
                with torch.no_grad():
                    for i in range(stage_idx):
                        emb_q_in, _ = adaptor.stages[i](emb_q_in)
                        emb_d_in, _ = adaptor.stages[i](emb_d_in)

                # 3. Current stage forward (trainable)
                emb_q_low, _ = stage(emb_q_in)
                emb_d_low, _ = stage(emb_d_in)

                # 4. Rank loss
                loss_r = rank_loss(emb_q_low, emb_d_low)

                # 5. Unsupervised loss (S-XBM)
                # We use query high-dim embedding vs its low-dim version with memory
                loss_u = sxbm.compute_unsup_loss(
                    emb_q_high, emb_q_low, adaptor, stage_idx
                )

                loss = loss_r + 1.0 * loss_u   # α=1.0

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # 6. Enqueue high-dim embeddings to memory (detached)
                sxbm.enqueue(emb_q_high.detach())
                sxbm.enqueue(emb_d_high.detach())

                total_rank += loss_r.item()
                total_unsup += loss_u.item()

                if batch_idx % 50 == 0:
                    print(f"  Epoch {epoch+1}, Batch {batch_idx}: "
                          f"Rank={loss_r.item():.4f}, Unsup={loss_u.item():.4f}")

            print(f"  Epoch {epoch+1} avg: Rank={total_rank/num_batches:.4f}, "
                  f"Unsup={total_unsup/num_batches:.4f}")

        print(f"Stage {stage_idx+1} done. Freezing.")

    print("\nTraining completed. SMEC adaptor ready for inference.")


if __name__ == "__main__":
    train()