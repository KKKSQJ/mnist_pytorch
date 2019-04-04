#coding:utf8
import os
from PIL import Image
from torch.utils import data
import numpy as np
from torchvision import transforms as T
from tqdm import tqdm


class Mnist(data.Dataset):
    def __init__(self, txt_root, transforms=None, train=True, test=False):
        self.test = test
        imgs = []
        with open(txt_root, 'r') as f:
            for line in f.readlines():
                line = line.strip()
                imgs.append(line)

        lens = len(imgs)
        np.random.seed(100)
        imgs = np.random.permutation(imgs)

        if self.test:
            self.imgs = imgs
        elif train:
            self.imgs = imgs[:int(0.8*lens)]
        else:
            self.imgs = imgs[int(0.8*lens):]

        if transforms is None:
            normalize = T.Normalize(
                mean=[0.5, 0.5, 0.5],
                std=[0.5, 0.5, 0.5]
            )

            if self.test or not train:
                self.transforms = T.Compose(
                    [
                        #T.Scale(28),
                        #T.CenterCrop(28),
                        T.ToTensor(),
                        #normalize,
                    ]
                )

            else:
                self.transforms = T.Compose(
                    [
                        #T.Scale(28),
                        #T.RandomSizedCrop(28),
                        #T.RandomHorizontalFlip(),
                        T.ToTensor(),
                        #normalize,
                    ]
                )

    def __getitem__(self, index):
        if self.test:
            img_path = self.imgs[index]
            label = self.imgs[index].strip().split('/')[-2]+"/"+self.imgs[index].strip().split('/')[-1]
        else:
            img_path = self.imgs[index].strip().split(" ")[0]
            label = int(self.imgs[index].strip().split(" ")[-1])

        data = Image.open(img_path)
        data = self.transforms(data)

        return data, label

    def __len__(self):
        return len(self.imgs)