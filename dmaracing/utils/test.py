import torch
import time


a = torch.ones((10000, 5, 70), device='cuda:0')
b = 2*torch.ones((10000, 5, 70), device='cuda:0')

t1 = time.time()

N = 100
for _ in range(N):
    a[:] = b
t2 = time.time()

print('no clone: ', (t2-t1)/N)

for _ in range(N):
    a = b.clone()
t2 = time.time()

print('clone: ', (t2-t1)/N)