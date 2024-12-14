from trainer import load_data
from trainer import get_device
def inspect_data(train_loader, probe_train_loader, val_loaders):
    print("=" * 50)
    print("Inspecting Training Data")
    print("=" * 50)
    for batch in train_loader:
        print("Batch Keys:", batch.keys())
        for key in batch.keys():
            print(f"{key} Shape: {batch[key].shape}")
        print("\nSample Data from Training Loader:")
        for key in batch.keys():
            print(f"{key}: {batch[key][0]}")  # 打印第一个样本的数据
        break  # 只打印一个批次，避免过多输出

    print("\n" + "=" * 50)
    print("Inspecting Probing Training Data")
    print("=" * 50)
    for batch in probe_train_loader:
        print("Batch Keys:", batch.keys())
        for key in batch.keys():
            print(f"{key} Shape: {batch[key].shape}")
        print("\nSample Data from Probe Training Loader:")
        for key in batch.keys():
            print(f"{key}: {batch[key][0]}")  # 打印第一个样本的数据
        break

    print("\n" + "=" * 50)
    print("Inspecting Validation Data (Normal)")
    print("=" * 50)
    for batch in val_loaders['normal']:
        print("Batch Keys:", batch.keys())
        for key in batch.keys():
            print(f"{key} Shape: {batch[key].shape}")
        print("\nSample Data from Validation Normal Loader:")
        for key in batch.keys():
            print(f"{key}: {batch[key][0]}")  # 打印第一个样本的数据
        break

    print("\n" + "=" * 50)
    print("Inspecting Validation Data (Wall)")
    print("=" * 50)
    for batch in val_loaders['wall']:
        print("Batch Keys:", batch.keys())
        for key in batch.keys():
            print(f"{key} Shape: {batch[key].shape}")
        print("\nSample Data from Validation Wall Loader:")
        for key in batch.keys():
            print(f"{key}: {batch[key][0]}")  # 打印第一个样本的数据
        break


# 调用 inspect_data 函数检查数据
device = get_device()
train_loader, probe_train_loader, val_loaders = load_data(device)
inspect_data(train_loader, probe_train_loader, val_loaders)
