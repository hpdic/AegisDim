import torch
import torch.nn as nn
import torch.nn.functional as F

class AdaptiveDimensionSelection(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.importance_logits = nn.Parameter(torch.zeros(input_dim))

    def forward(self, x, tau=1.0, hard=False, force_soft=False):
        # x: (B, input_dim)
        logits = self.importance_logits.unsqueeze(0).expand(x.size(0), -1)  # (B, D)

        if self.training or force_soft:
            # 训练时用软 Gumbel（即使 hard=True 也改为 soft，除非显式要求）
            weights = F.gumbel_softmax(logits, tau=tau, hard=False, dim=-1)
        else:
            # 纯推理：直接 top-k 硬掩码
            _, top_indices = torch.topk(self.importance_logits, self.output_dim)
            mask = torch.zeros_like(self.importance_logits)
            mask[top_indices] = 1.0
            weights = mask.unsqueeze(0).expand(x.size(0), -1)

        return weights


class SMECStage(nn.Module):
    def __init__(self, input_dim, output_dim, use_projection=True):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.use_projection = use_projection

        if use_projection:
            self.projection = nn.Sequential(
                nn.Linear(input_dim, input_dim),
                nn.LayerNorm(input_dim),
                nn.GELU(),
                nn.Linear(input_dim, input_dim)
            )
        else:
            self.projection = nn.Identity()

        self.ads = AdaptiveDimensionSelection(input_dim, output_dim)

    def forward(self, x, tau=1.0, hard=False, force_soft=False):
        residual = self.projection(x)
        adapted = x + residual

        mask = self.ads(adapted, tau, hard, force_soft)
        compressed = adapted * mask
        compressed = F.normalize(compressed, p=2, dim=-1)
        return compressed, mask


class SMECAdaptor(nn.Module):
    def __init__(self, stage_dims):
        super().__init__()
        self.stages = nn.ModuleList()
        for in_dim, out_dim in zip(stage_dims[:-1], stage_dims[1:]):
            self.stages.append(SMECStage(in_dim, out_dim))

    def forward(self, x, stage_idx=None, tau=1.0, hard=False, force_soft=False):
        if stage_idx is not None:
            return self.stages[stage_idx](x, tau, hard, force_soft)
        else:
            emb = x
            masks = []
            for stage in self.stages:
                emb, mask = stage(emb, tau, hard, force_soft)
                masks.append(mask)
            return emb, masks