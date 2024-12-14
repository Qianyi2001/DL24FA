import numpy as np
import os
import torch

# 数据路径
train_states_path = "/scratch/DL24FA/train/states.npy"
train_actions_path = "/scratch/DL24FA/train/actions.npy"
probe_train_path = "/scratch/DL24FA/probe_normal/train"
probe_val_normal_path = "/scratch/DL24FA/probe_normal/val"
probe_val_wall_path = "/scratch/DL24FA/probe_wall/val"

# 加载训练数据
print("=== Training Data ===")
train_states = np.load(train_states_path)  # Shape: (num_trajectories, trajectory_length, 2, 64, 64)
train_actions = np.load(train_actions_path)  # Shape: (num_trajectories, trajectory_length-1, 2)

print(f"States Shape: {train_states.shape}")  # 输出形状
print(f"Actions Shape: {train_actions.shape}")  # 输出形状

# 查看部分样本
print("\nSample Training State (Trajectory 0, Step 0):")
print(train_states[0, 0])  # 第一条轨迹的第一个状态

print("\nSample Training Action (Trajectory 0, Step 0):")
print(train_actions[0, 0])  # 第一条轨迹的第一个动作

# 加载探测训练数据
print("\n=== Probing Train Data ===")
probe_train_data = torch.load(probe_train_path)  # 假设探测数据保存为 PyTorch Tensor 格式
print(f"Probing Train Data Shape: {probe_train_data.shape}")

# 查看部分样本
print("\nSample Probing Train Data (Index 0):")
print(probe_train_data[0])

# 加载验证数据（normal）
print("\n=== Probing Validation Data (Normal) ===")
probe_val_normal_data = torch.load(probe_val_normal_path)  # 假设验证数据保存为 PyTorch Tensor 格式
print(f"Probing Validation Normal Data Shape: {probe_val_normal_data.shape}")

# 查看部分样本
print("\nSample Probing Validation Normal Data (Index 0):")
print(probe_val_normal_data[0])

# 加载验证数据（wall）
print("\n=== Probing Validation Data (Wall) ===")
probe_val_wall_data = torch.load(probe_val_wall_path)  # 假设验证数据保存为 PyTorch Tensor 格式
print(f"Probing Validation Wall Data Shape: {probe_val_wall_data.shape}")

# 查看部分样本
print("\nSample Probing Validation Wall Data (Index 0):")
print(probe_val_wall_data[0])
