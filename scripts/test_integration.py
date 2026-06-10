import torch
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.backbone import FrozenMistralBackbone
from models.smec_adaptor import SMECAdaptor

if __name__ == '__main__':
    print('Hardware Status Check Start')
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print('GPU is ONLINE. Using device:', torch.cuda.get_device_name(0))
    else:
        device = torch.device('cpu')
        print('Warning: GPU is OFFLINE. Falling back to CPU.')
    print('Hardware Status Check End\n')

    print('Step 1: Initializing Mistral 7B Backbone...')
    backbone = FrozenMistralBackbone()
    
    print('\nStep 2: Initializing SMEC Adaptor...')
    base_dim = 4096
    candidate_dims = [256, 512, 1024, 2048, 4096]
    adaptor = SMECAdaptor(base_dim=base_dim, candidate_dims=candidate_dims)
    adaptor = adaptor.to(device)

    print('\nStep 3: Feeding Real Text Stream...')
    texts = [
        'A cute dog playing in the grass.',
        'The intricate topological structure of high dimensional manifolds.'
    ]
    print('Input texts:', texts)

    embeddings = backbone(texts)
    print('Backbone output shape:', embeddings.shape)
    print('Backbone output device:', embeddings.device)

    embeddings = embeddings.to(device)

    print('\nStep 4: Running Adaptive Truncation...')
    truncated_output, selection_weights = adaptor(embeddings, tau=1.0, hard=True)

    print('Final truncated output shape:', truncated_output.shape)
    print('Selection weights matrix:')
    print(selection_weights)
    print('\nEnd to end integration test completed successfully.')