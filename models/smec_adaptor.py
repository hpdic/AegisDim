import torch
import torch.nn as nn
import torch.nn.functional as F

class AdaptiveDimensionSelection(nn.Module):
    def __init__(self, input_dim, candidate_dims):
        super(AdaptiveDimensionSelection, self).__init__()
        self.candidate_dims = candidate_dims
        self.num_candidates = len(candidate_dims)
        
        self.scorer = nn.Sequential(
            nn.Linear(input_dim, input_dim // 4),
            nn.ReLU(),
            nn.Linear(input_dim // 4, self.num_candidates)
        )

    def forward(self, x, tau=1.0, hard=True):
        logits = self.scorer(x)
        
        if self.training:
            weights = F.gumbel_softmax(logits, tau=tau, hard=hard, dim=1)
        else:
            idx = torch.argmax(logits, dim=1, keepdim=True)
            weights = torch.zeros_like(logits).scatter_(1, idx, 1.0)
            
        batch_size = x.size(0)
        device = x.device
        final_mask = torch.zeros(batch_size, x.size(1), device=device)
        
        for i, dim in enumerate(self.candidate_dims):
            mask = torch.zeros(1, x.size(1), device=device)
            mask[:, :dim] = 1.0
            final_mask += weights[:, i:i+1] * mask
            
        return final_mask, weights

class SMECAdaptor(nn.Module):
    def __init__(self, base_dim, candidate_dims):
        super(SMECAdaptor, self).__init__()
        self.base_dim = base_dim
        self.candidate_dims = sorted(candidate_dims)
        
        self.projection = nn.Sequential(
            nn.Linear(base_dim, base_dim),
            nn.LayerNorm(base_dim),
            nn.GELU(),
            nn.Linear(base_dim, base_dim)
        )
        
        self.ads = AdaptiveDimensionSelection(base_dim, self.candidate_dims)
        
    def forward(self, x, tau=1.0, hard=True):
        residual = self.projection(x)
        adapted_x = x + residual
        
        mask, selection_weights = self.ads(adapted_x, tau, hard)
        
        truncated_x = adapted_x * mask
        truncated_x = F.normalize(truncated_x, p=2, dim=1)
        
        return truncated_x, selection_weights