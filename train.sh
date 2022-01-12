CUDA_VISIBLE_DEVICES='3,4' python -m torch.distributed.launch --nproc_per_node 2 --master_port 1235  main2.py --cfg configs/swin_tiny_patch4_window7_224.yaml --data-path '.' --batch-size 12 --output covid19_416-4 --resume covid19_416-4/swin_large_416/default/ckpt_epoch_383.pth --base_lr 1e-6

CUDA_VISIBLE_DEVICES='3,4' python -m torch.distributed.launch --nproc_per_node 2 --master_port 1235  main2.py --cfg configs/swin4162.yaml --data-path '.' --batch-size 26 --output covid19_416-5

CUDA_VISIBLE_DEVICES='3,4' python -m torch.distributed.launch --nproc_per_node 2 --master_port 1235  main2.py --cfg configs/swin4163.yaml --data-path '.' --batch-size 12 --output covid19_416-6

CUDA_VISIBLE_DEVICES='3,4' python -m torch.distributed.launch --nproc_per_node 2 --master_port 1235  main2.py --cfg configs/swin4164.yaml --data-path '.' --batch-size 24 --output covid19_416-7