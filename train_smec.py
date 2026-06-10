import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from datasets import load_dataset
from collections import deque
import os
import sys
import math

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
    loss = F.softplus(diff)   # log(1+exp(diff)), numerically stable
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
        """Eq. 7: L1 distance between high-dim and low-dim similarities on hard samples."""
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
            mem_low = mem_high_flat
            for i in range(stage_idx):
                mem_low, _ = adaptor.stages[i](mem_low)
            mem_low, _ = adaptor.stages[stage_idx](mem_low)
        mem_low = mem_low.view(B, K, -1)  # (B, K, d)

        # High-dim similarities
        q_high_n = F.normalize(query_high, dim=1).unsqueeze(1)
        m_high_n = F.normalize(mem_high, dim=2)
        sim_high = (q_high_n * m_high_n).sum(dim=2)

        # Low-dim similarities
        q_low_n = F.normalize(query_low, dim=1).unsqueeze(1)
        m_low_n = F.normalize(mem_low, dim=2)
        sim_low = (q_low_n * m_low_n).sum(dim=2)

        loss = F.l1_loss(sim_low, sim_high.detach())
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

    # Warmup settings for unsup loss
    unsup_warmup_epochs = 3   # first 3 epochs only rank loss
    unsup_weight_start = 0.1
    unsup_weight_end = 1.0

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
            total_grad_norm = 0.0

            # Gumbel-Softmax temperature annealing
            tau = max(5.0 - epoch * 0.5, 1.0)  # decrease from 5.0 to 1.0

            # Unsupervised loss weight scheduling
            if epoch < unsup_warmup_epochs:
                alpha = 0.0
            else:
                # linearly increase from warmup to end
                progress = (epoch - unsup_warmup_epochs) / (epochs_per_stage - unsup_warmup_epochs)
                alpha = unsup_weight_start + (unsup_weight_end - unsup_weight_start) * progress

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

                # 3. Current stage forward (trainable) with annealed tau
                emb_q_low, _ = stage(emb_q_in, tau=tau, hard=(epoch >= unsup_warmup_epochs))
                emb_d_low, _ = stage(emb_d_in, tau=tau, hard=(epoch >= unsup_warmup_epochs))

                # 4. Rank loss
                loss_r = rank_loss(emb_q_low, emb_d_low)

                # 5. Unsupervised loss (S-XBM) - only after warmup and with weight alpha
                if alpha > 0 and sxbm.queue_tensor is not None and sxbm.queue_tensor.size(0) >= sxbm.topk:
                    loss_u = sxbm.compute_unsup_loss(emb_q_high, emb_q_low, adaptor, stage_idx)
                else:
                    loss_u = torch.tensor(0.0, device=device)

                loss = loss_r + alpha * loss_u

                optimizer.zero_grad()
                loss.backward()

                # Gradient clipping and logging
                total_norm = torch.nn.utils.clip_grad_norm_(stage.parameters(), max_norm=1.0)
                total_grad_norm += total_norm.item()

                optimizer.step()

                # 6. Enqueue high-dim embeddings to memory (detached)
                sxbm.enqueue(emb_q_high.detach())
                sxbm.enqueue(emb_d_high.detach())

                total_rank += loss_r.item()
                total_unsup += loss_u.item() if isinstance(loss_u, torch.Tensor) else loss_u

                if batch_idx % 50 == 0:
                    print(f"  Epoch {epoch+1}, Batch {batch_idx}: "
                          f"Rank={loss_r.item():.4f}, Unsup={loss_u.item():.4f}, "
                          f"tau={tau:.2f}, alpha={alpha:.2f}, grad_norm={total_norm:.2f}")

            avg_rank = total_rank / num_batches
            avg_unsup = total_unsup / num_batches
            avg_grad = total_grad_norm / num_batches
            print(f"  Epoch {epoch+1} avg: Rank={avg_rank:.4f}, Unsup={avg_unsup:.4f}, "
                  f"Grad={avg_grad:.2f}, tau={tau:.2f}, alpha={alpha:.2f}")

        print(f"Stage {stage_idx+1} done. Freezing.")

    print("\nTraining completed. SMEC adaptor ready for inference.")


if __name__ == "__main__":
    train()