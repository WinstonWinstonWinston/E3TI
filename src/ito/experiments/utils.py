import os
import glob
import datetime
import torch
from torch_geometric.data import Batch, Data


def get_latest_timestamp_directory(base_path):
    pattern = os.path.join(base_path, "train", "*")
    if not os.path.exists(base_path):
        print(f"Base path does not exist: {base_path}")
        return None
    directories = glob.glob(pattern)
    if not directories:
        print("No timestamp directories found.")
        return None
    latest_directory = max(directories, key=os.path.getmtime)
    print(f"Latest timestamp directory: {latest_directory}")
    return latest_directory

def batch_to_tensor(batch):
    poss = [data.x for data in batch.to_data_list()]
    return torch.stack(poss)


def batch_to_numpy(batch):
    return batch_to_tensor(batch).cpu().numpy()


def get_timestamp():
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")