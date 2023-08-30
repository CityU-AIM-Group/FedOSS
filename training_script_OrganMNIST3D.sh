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
--data_root='./datasets/MedMNIST/organmnist3d.npz' \
--lr=5e-4 \
--backbone='Resnet18_3D' \
--dataset='OrganMNIST3D' \
--known_class=7 \
--unknown_class=4 \
--seed=0 \
--batchsize=4 \
--epoches=100 \
--client_num=8 \
--worker_steps=1 \
--mode='Pretrain' \
--eps=1.0 \
--num_steps=100 \

python main.py \
--data_root='./datasets/MedMNIST/organmnist3d.npz' \
--lr=5e-4 \
--backbone='Resnet18_3D' \
--dataset='OrganMNIST3D' \
--known_class=7 \
--unknown_class=4 \
--seed=1 \
--batchsize=4 \
--epoches=100 \
--client_num=8 \
--worker_steps=1 \
--mode='Pretrain' \
--eps=1.0 \
--num_steps=100 \

python main.py \
--data_root='./datasets/MedMNIST/organmnist3d.npz' \
--lr=5e-4 \
--backbone='Resnet18_3D' \
--dataset='OrganMNIST3D' \
--known_class=7 \
--unknown_class=4 \
--seed=2 \
--batchsize=4 \
--epoches=100 \
--client_num=8 \
--worker_steps=1 \
--mode='Pretrain' \
--eps=1.0 \
--num_steps=100 \

python main.py \
--data_root='./datasets/MedMNIST/organmnist3d.npz' \
--lr=1e-4 \
--backbone='Resnet18_3D' \
--dataset='OrganMNIST3D' \
--known_class=7 \
--unknown_class=4 \
--seed=0 \
--batchsize=4 \
--epoches=30 \
--client_num=8 \
--worker_steps=1 \
--mode='Finetune' \
--eps=1. \
--num_steps=1 \
--unknown_weight=1.0 \
--dirichlet=0.5 \
--start_epoch='[5,10,15,20,25]' \
--sample_from=8 \

python main.py \
--data_root='./datasets/MedMNIST/organmnist3d.npz' \
--lr=1e-4 \
--backbone='Resnet18_3D' \
--dataset='OrganMNIST3D' \
--known_class=7 \
--unknown_class=4 \
--seed=1 \
--batchsize=4 \
--epoches=30 \
--client_num=8 \
--worker_steps=1 \
--mode='Finetune' \
--eps=1. \
--num_steps=1 \
--unknown_weight=1.0 \
--dirichlet=0.5 \
--start_epoch='[5,10,15,20,25]' \
--sample_from=8 \

python main.py \
--data_root='./datasets/MedMNIST/organmnist3d.npz' \
--lr=1e-4 \
--backbone='Resnet18_3D' \
--dataset='OrganMNIST3D' \
--known_class=7 \
--unknown_class=4 \
--seed=2 \
--batchsize=4 \
--epoches=30 \
--client_num=8 \
--worker_steps=1 \
--mode='Finetune' \
--eps=1. \
--num_steps=1 \
--unknown_weight=1.0 \
--dirichlet=0.5 \
--start_epoch='[5,10,15,20,25]' \
--sample_from=8 \

python main.py \
--data_root='./datasets/MedMNIST/organmnist3d.npz' \
--lr=5e-4 \
--backbone='Resnet18_3D' \
--dataset='OrganMNIST3D' \
--known_class=4 \
--unknown_class=7 \
--seed=0 \
--batchsize=4 \
--epoches=100 \
--client_num=8 \
--worker_steps=1 \
--mode='Pretrain' \
--eps=1.0 \
--num_steps=100 \

python main.py \
--data_root='./datasets/MedMNIST/organmnist3d.npz' \
--lr=5e-4 \
--backbone='Resnet18_3D' \
--dataset='OrganMNIST3D' \
--known_class=4 \
--unknown_class=7 \
--seed=1 \
--batchsize=4 \
--epoches=100 \
--client_num=8 \
--worker_steps=1 \
--mode='Pretrain' \
--eps=1.0 \
--num_steps=100 \

python main.py \
--data_root='./datasets/MedMNIST/organmnist3d.npz' \
--lr=5e-4 \
--backbone='Resnet18_3D' \
--dataset='OrganMNIST3D' \
--known_class=4 \
--unknown_class=7 \
--seed=2 \
--batchsize=4 \
--epoches=100 \
--client_num=8 \
--worker_steps=1 \
--mode='Pretrain' \
--eps=1.0 \
--num_steps=100 \

python main.py \
--data_root='./datasets/MedMNIST/organmnist3d.npz' \
--lr=1e-4 \
--backbone='Resnet18_3D' \
--dataset='OrganMNIST3D' \
--known_class=4 \
--unknown_class=7 \
--seed=0 \
--batchsize=4 \
--epoches=30 \
--client_num=8 \
--worker_steps=1 \
--mode='Finetune' \
--eps=1. \
--num_steps=1 \
--unknown_weight=1.0 \
--dirichlet=0.5 \
--start_epoch='[5,10,15,20,25]' \
--sample_from=8 \

python main.py \
--data_root='./datasets/MedMNIST/organmnist3d.npz' \
--lr=1e-4 \
--backbone='Resnet18_3D' \
--dataset='OrganMNIST3D' \
--known_class=4 \
--unknown_class=7 \
--seed=1 \
--batchsize=4 \
--epoches=30 \
--client_num=8 \
--worker_steps=1 \
--mode='Finetune' \
--eps=1. \
--num_steps=1 \
--unknown_weight=1.0 \
--dirichlet=0.5 \
--start_epoch='[5,10,15,20,25]' \
--sample_from=8 \

python main.py \
--data_root='./datasets/MedMNIST/organmnist3d.npz' \
--lr=1e-4 \
--backbone='Resnet18_3D' \
--dataset='OrganMNIST3D' \
--known_class=4 \
--unknown_class=7 \
--seed=2 \
--batchsize=4 \
--epoches=30 \
--client_num=8 \
--worker_steps=1 \
--mode='Finetune' \
--eps=1. \
--num_steps=1 \
--unknown_weight=1.0 \
--dirichlet=0.5 \
--start_epoch='[5,10,15,20,25]' \
--sample_from=8 \
