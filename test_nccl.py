import torch
import torch.distributed as dist

def main():
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    print(f"✅ Rank {rank} OK on {torch.cuda.get_device_name(torch.cuda.current_device())}")
    dist.destroy_process_group()

if __name__ == "__main__":
    main()
