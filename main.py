import numpy as np
import torch
import os

# a = torch.randn(2,3,4,5)
#
#
# b = a[:,2]
#
# c = torch.arange(0,32).reshape(2,2,8)
# print(c)
# index = torch.tensor([[[0,1,2],[3,4,5]],[[0,1,2],[3,4,5]]])
# print(index)
# e = torch.gather(c,dim=2,index=index)
# print(e)

mask = torch.arange(0,15).reshape(3, 5).float()
print(mask)
print(mask.detach())
print(mask.data)
print(mask.mean())