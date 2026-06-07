import torch
import numpy as np
import os

def test_sampler_fsdp_sharding():
    # In train.py:
    # sampler = torch.utils.data.SequentialSampler(range(_skip_samples, total_seqs))
    # train_loader = DataLoader(..., sampler=sampler, ...)

    # This sampler is NOT a DistributedSampler.
    # In FSDP, if all ranks use SequentialSampler, they all see the SAME data.

    world_size = 2
    total_seqs = 100

    # Simulated rank 0
    rank0_sampler = torch.utils.data.SequentialSampler(range(0, total_seqs))
    rank0_indices = list(rank0_sampler)

    # Simulated rank 1
    rank1_sampler = torch.utils.data.SequentialSampler(range(0, total_seqs))
    rank1_indices = list(rank1_sampler)

    if rank0_indices == rank1_indices:
        print("FAILURE: Both ranks see the same data. FSDP will duplicate work!")
    else:
        print("SUCCESS: Ranks see different data.")

def test_checkpointing_fsdp():
    # In train.py:
    # cp = { "model_state_dict": m.state_dict(), ... }
    # torch.save(cp, tmp)

    # In FSDP, state_dict() on a sharded model only returns the shard
    # unless special context is used.
    # Saving on all ranks without coordination will corrupt the file or
    # save partial data.

    print("ANALYSIS: train.py saves checkpoint without checking if rank == 0.")
    print("In multi-GPU, this will cause all 8 GPUs to write to the same file simultaneously.")

if __name__ == "__main__":
    test_sampler_fsdp_sharding()
    test_checkpointing_fsdp()
