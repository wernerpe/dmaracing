import torch
import time

#device = 'cuda:0'
device = 'cpu'



a = torch.rand((20000,20000), device = device)
b = torch.rand((20000,20000), device = device)
t0 = time.time()
for idx in range(20):
    if (idx % 10 )==0:
        print(idx)
    c = a*b
t1 = time.time()

print(t1-t0)