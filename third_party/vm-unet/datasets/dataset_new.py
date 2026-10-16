from torch.utils.data import Dataset
import numpy as np
import os
from PIL import Image
import random
import torch
from scipy import ndimage
from scipy.ndimage.interpolation import zoom

class NPY_datasets(Dataset):
    def __init__(self, path_Data, config, split='train'):
        super().__init__()
        self.split = split
        # Hangi dönüşümler kullanılacak?
        self.transformer = (
            config.train_transformer if split == 'train'
            else config.test_transformer
        )

        if split in ['train', 'val']:
            img_dir = os.path.join(path_Data, 'train', 'images')
            msk_dir = os.path.join(path_Data, 'train', 'masks')
            imgs = sorted(os.listdir(img_dir))
            msks = sorted(os.listdir(msk_dir))
            paired = list(zip(imgs, msks))

            # Aynı split reproducible olsun diye seed
            random.seed(config.seed)
            random.shuffle(paired)

            split_idx = int(0.8 * len(paired))
            if split == 'train':
                subset = paired[:split_idx]
            else:
                subset = paired[split_idx:]

            self.data = [
                [os.path.join(img_dir, i), os.path.join(msk_dir, m)]
                for i, m in subset
            ]

        elif split == 'test':
            img_dir = os.path.join(path_Data, 'val', 'images')
            msk_dir = os.path.join(path_Data, 'val', 'masks')
            imgs = sorted(os.listdir(img_dir))
            msks = sorted(os.listdir(msk_dir))
            self.data = [
                [os.path.join(img_dir, i), os.path.join(msk_dir, m)]
                for i, m in zip(imgs, msks)
            ]

        else:
            raise ValueError(f"Unknown split: {split}")

    def __getitem__(self, indx):
        img_path, msk_path = self.data[indx]
        img = np.array(Image.open(img_path).convert('RGB'))
        msk = np.expand_dims(
            np.array(Image.open(msk_path).convert('L')),
            axis=2
        ) / 255
        img, msk = self.transformer((img, msk))
        return img, msk

    def __len__(self):
        return len(self.data)


def random_rot_flip(image, label):
    k = np.random.randint(0, 4)
    image = np.rot90(image, k)
    label = np.rot90(label, k)
    axis = np.random.randint(0, 2)
    image = np.flip(image, axis=axis).copy()
    label = np.flip(label, axis=axis).copy()
    return image, label


def random_rotate(image, label):
    angle = np.random.randint(-20, 20)
    image = ndimage.rotate(image, angle, order=0, reshape=False)
    label = ndimage.rotate(label, angle, order=0, reshape=False)
    return image, label


class RandomGenerator(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        if random.random() > 0.5:
            image, label = random_rot_flip(image, label)
        elif random.random() > 0.5:
            image, label = random_rotate(image, label)
        x, y = image.shape
        if (x, y) != tuple(self.output_size):
            image = zoom(image, (self.output_size[0]/x, self.output_size[1]/y), order=3)
            label = zoom(label, (self.output_size[0]/x, self.output_size[1]/y), order=0)
        image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
        label = torch.from_numpy(label.astype(np.float32))
        sample = {'image': image, 'label': label.long()}
        return sample

