import torch
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.smec_adaptor import SMECAdaptor

if __name__ == '__main__':
    batch_size = 4
    base_dim = 4096
    candidate_dims = [256, 512, 1024, 2048, 4096]
    
    print('Initializing AegisDim SMEC Adaptor...')
    adaptor = SMECAdaptor(base_dim=base_dim, candidate_dims=candidate_dims)
    
    dummy_input = torch.randn(batch_size, base_dim)
    print('Input shape:', dummy_input.shape)
    
    truncated_output, selection_weights = adaptor(dummy_input, tau=1.0, hard=True)
    
    print('Output shape:', truncated_output.shape)
    print('Selection weights shape:', selection_weights.shape)
    print('Selection weights matrix:')
    print(selection_weights)
    print('Forward pass test completed successfully.')