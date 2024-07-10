import numpy as np
import h5py

import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


class BCDataset(Dataset):
    def __init__(self, hdf5_path, split='train', device="cuda"):
        super().__init__()
        self.device = device
        self.hdf5_file = h5py.File(hdf5_path, 'r', swmr=True, libver='latest')
        self.demos = [elem.decode("utf-8") for elem in np.array(self.hdf5_file["mask/{}".format(split)][:])]

        self._index_to_demo_id = dict()  # maps every index to a demo id
        self._demo_id_to_start_indices = dict()  # gives start index per demo id
        self._demo_id_to_demo_length = dict()

        # determine index mapping
        self.total_num = 0
        for ep in self.demos:
            demo_length = self.hdf5_file["data/{}".format(ep)].attrs["num_samples"]
            self._demo_id_to_start_indices[ep] = self.total_num
            self._demo_id_to_demo_length[ep] = demo_length

            for _ in range(demo_length):
                self._index_to_demo_id[self.total_num] = ep
                self.total_num += 1

        print("BCDataset: caching get_item calls...")
        self.getitem_cache = [self.getitem(i) for i in tqdm(range(len(self)))]

        self.hdf5_file.close()

    def __len__(self):
        return self.total_num
    
    def getitem(self, index):
        demo_id = self._index_to_demo_id[index]
        demo_start_index = self._demo_id_to_start_indices[demo_id]
        demo_length = self._demo_id_to_demo_length[demo_id]

        # start at offset index if not padding for frame stacking
        index_in_demo = index - demo_start_index

        # end at offset index if not padding for seq length
        end_index_in_demo = demo_length

        data = dict()

        # obs
        data["obs"] = dict()
        obs_keys = self.hdf5_file["data/{}/obs".format(demo_id)].keys()
        for k in obs_keys:
            data["obs"][k] = self.hdf5_file["data/{}/obs/{}".format(demo_id, k)][index_in_demo]

        # action
        data["action"] = self.hdf5_file["data/{}/actions".format(demo_id)][index_in_demo]

        # reward
        data["reward"] = self.hdf5_file["data/{}/rewards".format(demo_id)][index_in_demo]
        
        return data

    def __getitem__(self, index):
        return self.getitem_cache[index]
    
if __name__ == "__main__":
    test_path = "/cache1/kuangfuhang/tdmpc2/datasets/cup-catch/rgbd-3.hdf5"
    trainset = BCDataset(test_path)
    train_loader = DataLoader(
        dataset=trainset,
        batch_size=4,
        shuffle=True,
        num_workers=4,
    )
    for batch in train_loader:
        obs_dict = batch["obs"]
        assert isinstance(obs_dict, dict)
        action = batch["action"]
        assert action.shape[0] == 4
        print(action)