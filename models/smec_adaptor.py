import torch
import torch.nn as nn
import torch.nn.functional as F

class AdaptiveDimensionSelection(nn.Module):
    """ Gumbel-Softmax based dynamic dimension selection for a single reduction step.
        Input dim = D, output dim = D' (e.g. D/2).
        Learns importance logits for each dimension of the input.
    """
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        # 每个维度一个可学习 logit
        self.importance_logits = nn.Parameter(torch.zeros(input_dim))

    def forward(self, x, tau=1.0, hard=True):
        # x: (batch, input_dim)
        if self.training:
            # 重复 batch 次，使每个样本有独立的采样（Gumbel 噪声共享或独立？论文用 Gumbel-Softmax 近似 one-hot）
            # 这里对 batch 中每个样本使用相同的 importance_logits，但独立加 Gumbel 噪声
            logits = self.importance_logits.unsqueeze(0).expand(x.size(0), -1)  # (B, D)
            weights = F.gumbel_softmax(logits, tau=tau, hard=hard, dim=-1)    # (B, D), 近似 top-k one-hot
        else:
            # 推理时直接取 top-k
            _, top_indices = torch.topk(self.importance_logits, self.output_dim)
            mask = torch.zeros_like(self.importance_logits)
            mask[top_indices] = 1.0
            weights = mask.unsqueeze(0).expand(x.size(0), -1)  # (B, D)

        return weights  # 返回的是 soft mask (训练时) 或 binary mask (推理时)


class SMECStage(nn.Module):
    """ One sequential reduction stage: D -> D' with optional projection and ADS. """
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

    def forward(self, x, tau=1.0, hard=True):
        # 投影（残差）
        residual = self.projection(x)
        adapted = x + residual

        # 获取维度选择 mask
        mask = self.ads(adapted, tau, hard)       # (B, input_dim)
        compressed = adapted * mask               # 元素级乘法
        compressed = F.normalize(compressed, p=2, dim=-1)
        return compressed, mask


class SMECAdaptor(nn.Module):
    """ Full sequential adaptor: a list of stages for progressive reduction.
        Example: stages = [4096->2048, 2048->1024, 1024->512, 512->256]
    """
    def __init__(self, stage_dims):
        super().__init__()
        self.stages = nn.ModuleList()
        for in_dim, out_dim in zip(stage_dims[:-1], stage_dims[1:]):
            self.stages.append(SMECStage(in_dim, out_dim))

    def forward(self, x, stage_idx=None, tau=1.0, hard=True):
        """ If stage_idx is given, only run that stage. Otherwise run all sequentially. """
        if stage_idx is not None:
            return self.stages[stage_idx](x, tau, hard)
        else:
            emb = x
            masks = []
            for stage in self.stages:
                emb, mask = stage(emb, tau, hard)
                masks.append(mask)
            return emb, masks