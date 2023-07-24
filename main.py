import numpy as np
import torch
import os


a = torch.randn(2,3,4)
print(a)
b = a[:, 1, :]
print(b)
c = a[:, 1]
print(c)


