import numpy as np
import torch
import os


a = torch.randn(2,3,4)
b = a.max(dim=2)

print(a)
print(b)


