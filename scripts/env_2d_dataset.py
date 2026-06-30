import time
import numpy as np

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


if __name__ == "__main__":
    dataset = World2dDataset(LeRobotDataset("local/world2d", root="./world2d"))
    print(dataset[0])
    print(dataset[1])
