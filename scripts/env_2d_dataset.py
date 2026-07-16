import json
import os
import time
import numpy as np

import torch
from torch.utils.data import Dataset

from lerobot.datasets.lerobot_dataset import LeRobotDataset

class World2dDataset(Dataset):

    def __init__(self, dataset):
        self.dataset = dataset
        episode_indices = np.array(dataset.hf_dataset['episode_index'])
        episode_change = np.nonzero(np.diff(episode_indices))[0]
        # episode_change fires one step early (first data point is absorbed)
        self.episode_starts = np.concatenate([[0], episode_change + 1])
        # Exclusive
        self.episode_ends = np.concatenate([self.episode_starts[1:], [len(episode_indices)]])

        episode_lengths = self.episode_ends - self.episode_starts
        active_episodes = list(enumerate(episode_lengths))

        data_order = []
        episode_offsets = self.episode_starts.tolist()
        print("Reordering dataset...")
        t0 = time.monotonic()
        while len(active_episodes) > 0:
            remaining = []
            for episode_idx, remain_count in active_episodes:
                data_order.append(episode_offsets[episode_idx])
                episode_offsets[episode_idx] += 1
                if remain_count > 1:
                    remaining.append((episode_idx, remain_count - 1))
            active_episodes = remaining

        # Disgusting access patterns! woo!
        self.data_order = data_order
        assert len(self.data_order) == len(self.dataset)
        t1 = time.monotonic()
        print(f"Done. ({(t1 - t0)*1000:.2f}ms)")
    
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        frame = self.data_order[index]
        return self.dataset[frame]

class SmallPackedDataset(Dataset):

    def __init__(self, *, keys=None, data=None, root=None):
        if root is None:
            self.keys = keys
            self.data = data
        else:
            with open(os.path.join(root, "meta.json"), "r") as jf:
                self.keys = json.load(jf)

            self.data = []
            for i in range(len(self.keys)):
                self.data.append(np.load(os.path.join(root, f"{i}.npy")))

    def __len__(self):
        return len(self.data[0])

    def __getitem__(self, index):
        return {
            k: np.copy(self.data[i][index])
            for i, k in enumerate(self.keys)
        }
    
    def save(self, root):
        os.makedirs(root, exist_ok=True)
        with open(os.path.join(root, "meta.json"), "w") as jf:
            json.dump(self.keys, jf)
        for i, data in enumerate(self.data):
            np.save(os.path.join(root, f"{i}.npy"), data)

    @staticmethod
    def from_dataset(keys, dataset):
        data = [[] for _ in keys]
        for item in dataset:
            for i, k in enumerate(keys):
                data[i].append(item[k].numpy())
        for i, k in enumerate(keys):
            print(k, data[i][:10])
            data[i] = np.array(data[i], dtype=item[k].numpy().dtype)
        return SmallPackedDataset(keys=keys, data=data)

if __name__ == "__main__":
    dataset = World2dDataset(LeRobotDataset("local/world2d", root="./world2d"))

    keys = [
        'index',
        'episode_index',
        'frame_index',
        'observation.tokens',
        'observation.token_mask',
        'observation.token_categories',
        'action',
    ]
    import time
    import tqdm
    t0 = time.time()
    ds2 = SmallPackedDataset.from_dataset(keys, tqdm.tqdm(dataset))
    t1 = time.time()
    for item in tqdm.tqdm(ds2):
        pass
    t2 = time.time()
    print(t2 - t1, t1 - t0)
    ds2.save("world2d_reorder")
