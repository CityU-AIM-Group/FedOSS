#!/bin/bash
#SBATCH -J training
#SBATCH --gres=gpu:1
#SBATCH --partition=team1
#SBATCH -w node30
#SBATCH -c 2
#SBATCH -N 1

echo "Submitted from:"$SLURM_SUBMIT_DIR" on node:"$SLURM_SUBMIT_HOST
echo "Running on node "$SLURM_JOB_NODELIST 
echo "Allocate Gpu Units:"$CUDA_VISIBLE_DEVICES

nvidia-smi

python main.py \
--data_root='./datasets/MedMNIST/bloodmnist.npz' \
--lr=5e-4 \
--backbone='Resnet18' \
--dataset='Bloodmnist' \
--known_class=5 \
--unknown_class=3 \
--seed=0 \
--batchsize=8 \
--epoches=100 \
--client_num=16 \
--worker_steps=1 \
--mode='Pretain' \
--dirichlet=0.5 \

python main.py \
--data_root='./datasets/MedMNIST/bloodmnist.npz' \
--lr=5e-4 \
--backbone='Resnet18' \
--dataset='Bloodmnist' \
--known_class=5 \
--unknown_class=3 \
--seed=1 \
--batchsize=8 \
--epoches=100 \
--client_num=16 \
--worker_steps=1 \
--mode='Pretain' \
--dirichlet=0.5 \

python main.py \
--data_root='./datasets/MedMNIST/bloodmnist.npz' \
--lr=5e-4 \
--backbone='Resnet18' \
--dataset='Bloodmnist' \
--known_class=5 \
--unknown_class=3 \
--seed=2 \
--batchsize=8 \
--epoches=100 \
--client_num=16 \
--worker_steps=1 \
--mode='Pretain' \
--dirichlet=0.5 \

python main.py \
--data_root='./datasets/MedMNIST/bloodmnist.npz' \
--lr=1e-4 \
--backbone='Resnet18' \
--dataset='Bloodmnist' \
--known_class=5 \
--unknown_class=3 \
--seed=0 \
--batchsize=8 \
--epoches=30 \
--client_num=16 \
--worker_steps=1 \
--mode='Finetune' \
--eps=0.1 \
--num_steps=1 \
--unknown_weight=1. \
--dirichlet=0.5 \
--start_epoch='[5,10,15,20,25]'
--sample_from=8 \

python main.py \
--data_root='./datasets/MedMNIST/bloodmnist.npz' \
--lr=1e-4 \
--backbone='Resnet18' \
--dataset='Bloodmnist' \
--known_class=5 \
--unknown_class=3 \
--seed=1 \
--batchsize=8 \
--epoches=30 \
--client_num=16 \
--worker_steps=1 \
--mode='Finetune' \
--eps=0.1 \
--num_steps=1 \
--unknown_weight=1. \
--dirichlet=0.5 \
--start_epoch='[5,10,15,20,25]'
--sample_from=8 \

python main.py \
--data_root='./datasets/MedMNIST/bloodmnist.npz' \
--lr=1e-4 \
--backbone='Resnet18' \
--dataset='Bloodmnist' \
--known_class=5 \
--unknown_class=3 \
--seed=2 \
--batchsize=8 \
--epoches=30 \
--client_num=16 \
--worker_steps=1 \
--mode='Finetune' \
--eps=0.1 \
--num_steps=1 \
--unknown_weight=1. \
--dirichlet=0.5 \
--start_epoch='[5,10,15,20,25]'
--sample_from=8 \


python main.py \
--data_root='./datasets/MedMNIST/bloodmnist.npz' \
--lr=5e-4 \
--backbone='Resnet18' \
--dataset='Bloodmnist' \
--known_class=3 \
--unknown_class=5 \
--seed=0 \
--batchsize=8 \
--epoches=100 \
--client_num=16 \
--worker_steps=1 \
--mode='Pretain' \
--dirichlet=0.5 \

python main.py \
--data_root='./datasets/MedMNIST/bloodmnist.npz' \
--lr=5e-4 \
--backbone='Resnet18' \
--dataset='Bloodmnist' \
--known_class=3 \
--unknown_class=5 \
--seed=1 \
--batchsize=8 \
--epoches=100 \
--client_num=16 \
--worker_steps=1 \
--mode='Pretain' \
--dirichlet=0.5 \

python main.py \
--data_root='./datasets/MedMNIST/bloodmnist.npz' \
--lr=5e-4 \
--backbone='Resnet18' \
--dataset='Bloodmnist' \
--known_class=3 \
--unknown_class=5 \
--seed=2 \
--batchsize=8 \
--epoches=100 \
--client_num=16 \
--worker_steps=1 \
--mode='Pretain' \
--dirichlet=0.5 \

python main.py \
--data_root='./datasets/MedMNIST/bloodmnist.npz' \
--lr=1e-4 \
--backbone='Resnet18' \
--dataset='Bloodmnist' \
--known_class=3 \
--unknown_class=5 \
--seed=0 \
--batchsize=8 \
--epoches=30 \
--client_num=16 \
--worker_steps=1 \
--mode='Finetune' \
--eps=0.1 \
--num_steps=1 \
--unknown_weight=1. \
--dirichlet=0.5 \
--start_epoch='[5,10,15,20,25]'
--sample_from=8 \

python main.py \
--data_root='./datasets/MedMNIST/bloodmnist.npz' \
--lr=1e-4 \
--backbone='Resnet18' \
--dataset='Bloodmnist' \
--known_class=3 \
--unknown_class=5 \
--seed=1 \
--batchsize=8 \
--epoches=30 \
--client_num=16 \
--worker_steps=1 \
--mode='Finetune' \
--eps=0.1 \
--num_steps=1 \
--unknown_weight=1. \
--dirichlet=0.5 \
--start_epoch='[5,10,15,20,25]'
--sample_from=8 \

python main.py \
--data_root='./datasets/MedMNIST/bloodmnist.npz' \
--lr=1e-4 \
--backbone='Resnet18' \
--dataset='Bloodmnist' \
--known_class=3 \
--unknown_class=5 \
--seed=2 \
--batchsize=8 \
--epoches=30 \
--client_num=16 \
--worker_steps=1 \
--mode='Finetune' \
--eps=0.1 \
--num_steps=1 \
--unknown_weight=1. \
--dirichlet=0.5 \
--start_epoch='[5,10,15,20,25]'
--sample_from=8 \