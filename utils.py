import torch
import torch.nn as nn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def count_parameters(model: nn.Module):
    return sum(p.numel() for p in model.parameters())
