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
--data_root='./datasets/Hyperkvasir_processed/labeled-images/' \
--lr=5e-4 \
--backbone='Resnet18' \
--dataset='Hyperkvasir' \
--known_class=6 \
--unknown_class=3 \
--seed=0 \
--batchsize=8 \
--epoches=200 \
--client_num=8 \
--worker_steps=1 \
--mode='DUS_baseline' \

python main.py \
--data_root='./datasets/Hyperkvasir_processed/labeled-images/' \
--lr=5e-4 \
--backbone='Resnet18' \
--dataset='Hyperkvasir' \
--known_class=6 \
--unknown_class=3 \
--seed=1 \
--batchsize=8 \
--epoches=200 \
--client_num=8 \
--worker_steps=1 \
--mode='DUS_baseline' \

python main.py \
--data_root='./datasets/Hyperkvasir_processed/labeled-images/' \
--lr=5e-4 \
--backbone='Resnet18' \
--dataset='Hyperkvasir' \
--known_class=6 \
--unknown_class=3 \
--seed=2 \
--batchsize=8 \
--epoches=200 \
--client_num=8 \
--worker_steps=1 \
--mode='DUS_baseline' \

python main.py \
--data_root='./datasets/Hyperkvasir_processed/labeled-images/' \
--lr=1e-4 \
--backbone='Resnet18' \
--dataset='Hyperkvasir' \
--known_class=6 \
--unknown_class=3 \
--seed=0 \
--batchsize=8 \
--epoches=30 \
--client_num=8 \
--worker_steps=1 \
--mode='DUS_CUS_finetune' \
--num_steps=5 \
--unknown_weight=1. \
--sample_from=8 \
--start_epoch='[5,10,15,20,25]'

python main.py \
--data_root='./datasets/Hyperkvasir_processed/labeled-images/' \
--lr=1e-4 \
--backbone='Resnet18' \
--dataset='Hyperkvasir' \
--known_class=6 \
--unknown_class=3 \
--seed=1 \
--batchsize=8 \
--epoches=30 \
--client_num=8 \
--worker_steps=1 \
--mode='DUS_CUS_finetune' \
--num_steps=5 \
--unknown_weight=1. \
--sample_from=8 \
--start_epoch='[5,10,15,20,25]'

python main.py \
--data_root='./datasets/Hyperkvasir_processed/labeled-images/' \
--lr=1e-4 \
--backbone='Resnet18' \
--dataset='Hyperkvasir' \
--known_class=6 \
--unknown_class=3 \
--seed=2 \
--batchsize=8 \
--epoches=30 \
--client_num=8 \
--worker_steps=1 \
--mode='DUS_CUS_finetune' \
--num_steps=5 \
--unknown_weight=1. \
--sample_from=8 \
--start_epoch='[5,10,15,20,25]'


python main.py \
--data_root='./datasets/Hyperkvasir_processed/labeled-images/' \
--lr=5e-4 \
--backbone='Resnet18' \
--dataset='Hyperkvasir' \
--known_class=6 \
--unknown_class=9 \
--seed=0 \
--batchsize=8 \
--epoches=200 \
--client_num=8 \
--worker_steps=1 \
--mode='DUS_baseline' \

python main.py \
--data_root='./datasets/Hyperkvasir_processed/labeled-images/' \
--lr=5e-4 \
--backbone='Resnet18' \
--dataset='Hyperkvasir' \
--known_class=6 \
--unknown_class=9 \
--seed=1 \
--batchsize=8 \
--epoches=200 \
--client_num=8 \
--worker_steps=1 \
--mode='DUS_baseline' \

python main.py \
--data_root='./datasets/Hyperkvasir_processed/labeled-images/' \
--lr=5e-4 \
--backbone='Resnet18' \
--dataset='Hyperkvasir' \
--known_class=6 \
--unknown_class=9 \
--seed=2 \
--batchsize=8 \
--epoches=200 \
--client_num=8 \
--worker_steps=1 \
--mode='DUS_baseline' \


python main.py \
--data_root='./datasets/Hyperkvasir_processed/labeled-images/' \
--lr=1e-4 \
--backbone='Resnet18' \
--dataset='Hyperkvasir' \
--known_class=6 \
--unknown_class=9 \
--seed=0 \
--batchsize=8 \
--epoches=30 \
--client_num=8 \
--worker_steps=1 \
--mode='DUS_CUS_finetune' \
--num_steps=5 \
--unknown_weight=1. \
--sample_from=8 \
--start_epoch='[5,10,15,20,25]'

python main.py \
--data_root='./datasets/Hyperkvasir_processed/labeled-images/' \
--lr=1e-4 \
--backbone='Resnet18' \
--dataset='Hyperkvasir' \
--known_class=6 \
--unknown_class=9 \
--seed=1 \
--batchsize=8 \
--epoches=30 \
--client_num=8 \
--worker_steps=1 \
--mode='DUS_CUS_finetune' \
--num_steps=5 \
--unknown_weight=1. \
--sample_from=8 \
--start_epoch='[5,10,15,20,25]'

python main.py \
--data_root='./datasets/Hyperkvasir_processed/labeled-images/' \
--lr=1e-4 \
--backbone='Resnet18' \
--dataset='Hyperkvasir' \
--known_class=6 \
--unknown_class=9 \
--seed=2 \
--batchsize=8 \
--epoches=30 \
--client_num=8 \
--worker_steps=1 \
--mode='DUS_CUS_finetune' \
--num_steps=5 \
--unknown_weight=1. \
--sample_from=8 \
--start_epoch='[5,10,15,20,25]'
