import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn

from muon import SingleDeviceMuon

dim = 128

model = nn.Sequential(*[nn.Linear(dim, dim, bias=False) for _ in range(8)])
cross_entropy = nn.CrossEntropyLoss()

def batch(size, dim):
    data = torch.normal(mean=0, std=1, size=(size, dim))
    target = torch.randint(low=0, high=dim, size=(size,))
    return data, target

def print_momentum_buffer(optimizer):
    print('momentum_buffer for parameter id:')
    for parameter_id, s in optimizer.state_dict()['state'].items():
        print(f'{parameter_id}:', f'{s['momentum_buffer'].shape} tensor' if 'momentum_buffer' in s else None)

optimizer = SingleDeviceMuon(model.parameters())

data, target = batch(1, dim)
loss = cross_entropy(model(data), target)
loss.backward()
optimizer.step()
optimizer.zero_grad()

print_momentum_buffer(optimizer)
