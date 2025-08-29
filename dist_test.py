import os

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn

from muon import Muon

def batch(size, dim):
    data = torch.normal(mean=0, std=1, size=(size, dim))
    target = torch.randint(low=0, high=dim, size=(size,))
    return data, target

def print_momentum_buffer(state_dict):
    print('momentum_buffer for parameter id:')
    for parameter_id, s in state_dict['state'].items():
        print(f'{parameter_id}:', f'{s['momentum_buffer'].shape} tensor' if 'momentum_buffer' in s else None)

def is_primary(rank):
    return not rank

def compare_params(init_data, params):
    for i, (init_d, p) in enumerate(zip(init_data, params)):
        if torch.equal(init_d, p.data):
            print('Parameter id', i, 'unchanged since init!')

def main():

    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ngpus_per_node = torch.cuda.device_count()
    world_size *= ngpus_per_node

    mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, world_size))

def main_worker(gpu, ngpus_per_node, world_size):

    backend = 'nccl'
    dist_url = 'env://'
    rank = int(os.environ["RANK"])
    rank = rank * ngpus_per_node + gpu
    os.environ['CUDA_VISIBLE_DEVICES'] = str(rank)

    dist.init_process_group(
        backend=backend,
        init_method=dist_url,
        world_size=world_size,
        rank=rank
    )

    dim = 128
    model = nn.Sequential(*[nn.Linear(dim, dim, bias=False) for _ in range(world_size)])
    cross_entropy = nn.CrossEntropyLoss()
    model.cuda()
    cross_entropy.cuda()

    model = torch.nn.parallel.DistributedDataParallel(model)

    optimizer = Muon(list(model.parameters()))
    if is_primary(rank):
        init_data = [p.data.clone() for p in model.parameters()]

    data, target = batch(1, dim)
    loss = cross_entropy(model(data.cuda()), target.cuda())
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    if is_primary(rank):
        print('Without state dict pre-hook to gather:')
        print_momentum_buffer(optimizer.state_dict())
        print()
        compare_params(init_data, model.parameters())
        print()

    def gather(optimizer):
        for group in optimizer.param_groups:
            params = group["params"]
            buffers = [optimizer.state[p].get("momentum_buffer", torch.zeros_like(p)) for p in params]
            buffers.extend([torch.empty_like(params[-1])] * (world_size - len(params) % world_size))
            for base_i in range(len(params))[::world_size]:
                dist.gather(buffers[base_i + rank], gather_list=buffers[base_i:base_i + world_size] if is_primary(rank) else None)

    hook = optimizer.register_state_dict_pre_hook(gather)

    data, target = batch(1, dim)
    loss = cross_entropy(model(data.cuda()), target.cuda())
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    state_dict = optimizer.state_dict()
    if is_primary(rank):
        print('With state dict pre-hook to gather:')
        print_momentum_buffer(state_dict)
        print()
        compare_params(init_data, model.parameters())
        print()

    hook.remove()

    def actually_gather(optimizer):
        for group in optimizer.param_groups:
            params = group["params"]
            assigned_b = optimizer.state[params[rank]]["momentum_buffer"]
            buffer = None
            if is_primary(rank):
                buffer = [optimizer.state[p].get("momentum_buffer", torch.zeros_like(p)) for p in params]
            dist.gather(assigned_b, gather_list=buffer)
            if is_primary(rank):
                for p, b in zip(params, buffer):
                    optimizer.state[p]["momentum_buffer"] = b

    hook = optimizer.register_state_dict_pre_hook(actually_gather)

    data, target = batch(1, dim)
    loss = cross_entropy(model(data.cuda()), target.cuda())
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    state_dict = optimizer.state_dict()
    if is_primary(rank):
        print('With state dict pre-hook to actually gather:')
        print_momentum_buffer(state_dict)
        print()
        compare_params(init_data, model.parameters())
        print()
    dist.destroy_process_group()

if __name__ == '__main__':
    main()
