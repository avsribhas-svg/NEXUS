import os
import glob
import numpy as np
import torch
from torch_geometric.data import Data, InMemoryDataset

CHANNEL_MAXES = torch.tensor([50.0, 30.0, 20.0, 10.0, 15.0, 30.0, 10.0, 10.0], dtype=torch.float32)
GJ_MAX = 50.0

class BioelectricDataset(InMemoryDataset):
    def __init__(self, root, split="train", transform=None, pre_transform=None):
        self.split = split
        super().__init__(root, transform, pre_transform)
        self.load(self.processed_paths[0])

    @property
    def raw_dir(self):        return os.path.join(self.root, self.split)
    @property
    def raw_file_names(self): return []
    @property
    def processed_dir(self):  return os.path.join(self.root, "processed")
    @property
    def processed_file_names(self): return [f"{self.split}.pt"]

    def download(self):
        pass

    def process(self):
        paths = sorted(glob.glob(os.path.join(self.root, self.split, "*.npz")))
        data_list = []
        for path in paths:
            z = np.load(path, allow_pickle=True)

            x = torch.tensor(np.asarray(z["channel_densities"]), dtype=torch.float32) / CHANNEL_MAXES
            edge_index = torch.tensor(np.asarray(z["edge_index"]), dtype=torch.long)
            edge_attr = torch.tensor(np.asarray(z["conductances"]), dtype=torch.float32).reshape(-1, 1) / GJ_MAX
            y = torch.tensor(np.asarray(z["vmem_steady_state"]), dtype=torch.float32)
            pos = torch.tensor(np.asarray(z["cell_positions"]), dtype=torch.float32)

            data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y, pos=pos)
            data.config_id = str(z["config_id"])
            data.is_perturbation = bool(z["is_perturbation"])
            data_list.append(data)

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]
        self.save(data_list, self.processed_paths[0])
