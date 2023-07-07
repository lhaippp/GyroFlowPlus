import torch

import numpy as np

from torch.utils.data import ConcatDataset
from torch.utils.data import Dataset, DataLoader

from transform import transforms_lib


class TestDataset(Dataset):
    def __init__(self, benchmark_path, input_transform):
        self.input_transform = input_transform

        self.samples = np.load(benchmark_path, allow_pickle=True)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        imgs = [self.samples[idx]["img1"], self.samples[idx]["img2"]]

        gyro_homo = self.samples[idx]["homo"]

        gt_flow = self.samples[idx]["gt_flow"]
        gt_flow = torch.from_numpy(gt_flow).permute(2, 0, 1)

        split = self.samples[idx]["split"]

        gyro_filed = transforms_lib.homo_to_flow(np.expand_dims(gyro_homo, 0), H=600, W=800)[0]

        gyro_filed = gyro_filed.squeeze()

        if self.input_transform:
            imgs_it = [self.input_transform(i) for i in imgs]

        ret = {"img{}".format(i + 1): v for i, v in enumerate(imgs_it)}

        ret["gyro_field"] = gyro_filed
        ret["gt_flow"] = gt_flow
        ret["label"] = split
        return ret


class GHOFTestDataset(TestDataset):
    def __init__(self, benchmark_path, input_transform):
        super(GHOFTestDataset, self).__init__(benchmark_path, input_transform)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        split = self.samples[idx]["split"]

        imgs = [self.samples[idx]["img1"], self.samples[idx]["img2"]]

        gyro_homo = self.samples[idx]["homo"]

        gt_flow = self.samples[idx]["gt_flow"]
        gt_flow = torch.from_numpy(gt_flow).permute(2, 0, 1)

        homo_gt = self.samples[idx]["homo_field"]
        homo_gt = torch.from_numpy(homo_gt).permute(2, 0, 1)

        gyro_filed = transforms_lib.homo_to_flow(np.expand_dims(gyro_homo, 0), H=600, W=800)[0]

        gyro_filed = gyro_filed.squeeze()

        if self.input_transform:
            imgs_it = [self.input_transform(i) for i in imgs]

        ret = {"img{}".format(i + 1): v for i, v in enumerate(imgs_it)}

        ret["gyro_field"] = gyro_filed
        ret["gt_flow"] = gt_flow
        ret["label"] = split
        ret["homo_field"] = homo_gt
        return ret


def fetch_dataloader(types, status_manager):
    """
    Fetches the DataLoader object for each type in types from data_dir.

    Args:
        types: (list) has one or more of 'train', 'val', 'test' depending on which data is required
        status_manager: (class) status_manager

    Returns:
        data: (dict) contains the DataLoader object for each type in types
    """
    dataloaders = {}
    input_transform = transforms_lib.fetch_input_transform()

    for split in ['train', 'valid', 'test']:
        if split in types:
            benchmark_path_gof_clean = 'GHOF_Clean_20230705.npy'
            benchmark_path_gof_final = 'GHOF_Final_20230705.npy'

            if split == "test":
                input_transform = transforms_lib.fetch_input_transform(if_normalize=False)
                ds = ConcatDataset([
                    GHOFTestDataset(benchmark_path_gof_clean, input_transform=input_transform),
                    TestDataset(benchmark_path_gof_final, input_transform=input_transform)
                ])
                dl = [
                    DataLoader(s,
                               batch_size=1,
                               shuffle=False,
                               num_workers=status_manager.params.num_workers,
                               pin_memory=status_manager.params.cuda) for s in ds.datasets
                ]
            else:
                raise Exception()

            dataloaders[split] = dl
    return dataloaders
