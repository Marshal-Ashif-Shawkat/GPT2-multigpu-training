
# Learning resource: https://youtu.be/l8pRSuU81PU?si=rW9Agi6va0dib4SL

import torch
from torch import nn
import torch.nn.functional as F
from dataclasses import dataclass
import numpy as np
import random
from tqdm import tqdm
import os
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

from model import config, GPT

torch.set_float32_matmul_precision('high')

init_process_group(backend='nccl')
ddp_rank = int(os.environ['RANK'])
ddp_local_rank = int(os.environ['LOCAL_RANK'])
ddp_world_size = int(os.environ['WORLD_SIZE'])
device = f'cuda:{ddp_local_rank}'
torch.cuda.set_device(device)
master_process = (ddp_rank == 0) # this process will do logging, checkpointing etc.

def set_seed(SEED):
    torch.manual_seed(SEED)
    random.seed(SEED)
    np.random.seed(SEED)

set_seed(SEED=42)

config = config()
if master_process:
    print("vocab size")
    print(config.vocab_size)
model = GPT(config)
model = model.to(device)
model = torch.compile(model)
model = DDP(model, device_ids=[ddp_local_rank])
num_params = sum(p.numel() for p in model.parameters())


class SimpleDataLoader:
    def __init__(self, batch_size, config, filepath, process_rank, num_processes):
        self.batch_size = batch_size
        self.config = config
        self.filepath = filepath
        self.numel = self.batch_size * self.config.block_size # no of tokens in a batch
        self.process_rank = process_rank
        self.num_processes = num_processes
        self.tokens = self._load_tokens()
        self.current_position = self.batch_size * self.config.block_size * self.process_rank
    
    def reset(self):
        self.current_position = random.randint(0, len(self.tokens) - (self.numel + 1))

    def _load_tokens(self):
        npt = np.load(self.filepath)
        ptt = torch.tensor(npt, dtype=torch.long)
        return ptt

    def __len__(self):
        return len(self.tokens) // self.numel
        
    def next_batch(self):
        if self.current_position + self.numel + 1 > len(self.tokens):
            self.reset()
        buffer = self.tokens[self.current_position: (self.current_position+self.numel+1)]
        self.current_position += (self.numel * self.num_processes)
        
        B, T = self.batch_size, self.config.block_size
        input_tokens = buffer[:-1].reshape(B, T)
        target_tokens = buffer[1:].reshape(B, T)
        return input_tokens, target_tokens




BATCH_SIZE = 32

train_path = "train_tokens_wikitext_2.npy"
test_path = "test_tokens_wikitext_2.npy"
train_loader = SimpleDataLoader(BATCH_SIZE, config, train_path, process_rank=ddp_rank, num_processes=ddp_world_size)
test_loader = SimpleDataLoader(BATCH_SIZE, config, test_path, process_rank=ddp_rank, num_processes=ddp_world_size)
if master_process:
    print(f"No of batches in training: {len(train_loader)}")
    print(f"No of batches in testing: {len(test_loader)}")




def train_step(dataloader, device, loss_fn, optimizer, ddp_world_size):
    batch_loss = 0
    model.train()
    nsteps = len(dataloader)//ddp_world_size
    for _ in range(nsteps):
        inputs, targets = dataloader.next_batch()
        inputs, targets = inputs.to(device), targets.to(device)
        with torch.autocast(device_type=device, dtype=torch.bfloat16):
            logits = model(inputs)
            # CrossEntropyLoss takes 2D tensor as input
            loss = loss_fn(logits.flatten(0,1), targets.flatten(0,1))
        batch_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return batch_loss / nsteps

@torch.no_grad()
def test_step(dataloader, device, loss_fn, ddp_world_size):
    batch_loss = 0
    model.eval()
    nsteps = len(dataloader)//ddp_world_size
    for _ in range(nsteps):
        inputs, targets = dataloader.next_batch()
        inputs, targets = inputs.to(device), targets.to(device)
        with torch.autocast(device_type=device, dtype=torch.bfloat16):
            logits = model(inputs)
            # CrossEntropyLoss takes 2D tensor as input
            loss = loss_fn(logits.flatten(0,1), targets.flatten(0,1))
        batch_loss += loss.item()

    return batch_loss / nsteps
    


NUM_EPOCH = 2

loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=6e-4, weight_decay=0.1, fused=True)

if master_process:
    data_iter = tqdm(range(NUM_EPOCH), desc=f"Epoch")
else:
    data_iter = range(NUM_EPOCH)
    
for epoch_id in data_iter:
    train_loss = train_step(train_loader, device, loss_fn, optimizer, ddp_world_size)
    test_loss = test_step(test_loader, device, loss_fn, ddp_world_size)
    train_loader.reset()
    test_loader.current_position = 0
    if master_process:
        print(f"Epoch: {epoch_id+1:>3} | train_loss: {train_loss:.4f} | test_loss: {test_loss:.4f}")
        if epoch_id % 10:
            torch.save(model.module.state_dict(), "gpt2_wikitext2.pt")


destroy_process_group()


