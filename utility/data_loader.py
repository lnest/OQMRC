# -*- coding: utf-8 -*-
# ------------------------------------
# Create On 2018/9/19
# File Name: data_loader
# Edit Author: lnest
# ------------------------------------
import json
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader


class SequenceDataset(Dataset):
    def __init__(self, file_name, transform=None):
        self.json_file = file_name
        self.data = self.__load_data()
        self.transform = transform

    def __load_data(self):
        with open(self.json_file) as jfr:
            data = json.load(jfr)
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        if self.transform:
            sample = self.transform(sample)
        return sample


class ToTensor(object):
    def __call__(self, sample):
        passage, query, query_id = sample['passage'], sample['query'], sample['query_id']
        return {'passage': torch.tensor(passage, dtype=torch.long),
                'query': torch.tensor(query, dtype=torch.long),
                'query_id': torch.tensor(query_id, dtype=torch.long),
                'answer': torch.tensor(sample['answer'], dtype=torch.long)}


def get_dataloader(json_file='./data/dl_train.json', batch_size=256, num_workers=4, shuffle=True):
    sequence_dataset = SequenceDataset(json_file, transform=transforms.Compose([ToTensor()]))
    dataloader = DataLoader(sequence_dataset, batch_size=batch_size,
                            shuffle=shuffle, num_workers=num_workers)
    return dataloader
