import glob
from typing import List

import cv2
import torch
from torch.utils.data import ConcatDataset
from torchvision import datasets, transforms
import torch.utils.data as data
import os
from PIL import Image
import numpy as np
from torch.utils.data.sampler import SubsetRandomSampler, SequentialSampler

base_transform = transforms.Compose([
    transforms.ToTensor(),
    # transforms.Normalize([0.5] * 3, [0.5] * 3)
])

norm_transform = transforms.Compose([
    transforms.Normalize([0.5] * 3, [0.5] * 3)
])

inverse_transform = transforms.Compose([transforms.Normalize(mean=[0., 0., 0.],
                                                             std=[1 / 0.5] * 3),
                                        transforms.Normalize(mean=[-0.5] * 3,
                                                             std=[1., 1., 1.]),
                                        ])

upscale_factor = 4


def image_loader(path):
    img = Image.open(path)
    img_tensor = base_transform(img)
    return img_tensor


def lr_image_loader(path, upscale_factor):
    """ Downscale an HR image to an LR image. """
    img = Image.open(path)
    width, height = img.size
    tw = width // upscale_factor
    th = height // upscale_factor
    # print((tw,th))
    img = img.resize(size=(tw, th), resample=Image.BILINEAR)
    # print("lr:",img.size)
    img_tensor = base_transform(img)
    # print(img_tensor.shape)
    return img_tensor


# horizontally stacked images to list of images
# 768x256 -> [256x256, 256x256, 256x256]
def unhstack_sept(hstack_img: np.ndarray) -> List[np.ndarray]:
    h, w, c = hstack_img.shape
    imglist = [hstack_img[:, i * h:((i + 1) * h), :] for i in range(0, int(w // h))]
    return imglist


# reads horizontally stacked images to list of images
# 768x256 -> [256x256, 256x256, 256x256]
def read_sept(img_path: str) -> List[np.ndarray]:
    hstack_img = cv2.imread(img_path)
    assert hstack_img is not None, f'cant read image {img_path}'
    hstack_img = cv2.cvtColor(hstack_img, cv2.COLOR_BGR2RGB)
    return unhstack_sept(hstack_img)


class IMDBDataset(data.Dataset):
    def __init__(self, hr_dir, upscale_factor, hrheight):
        self.file_hr_dir = hr_dir
        self.transform = base_transform
        self.upscale_factor = upscale_factor
        self.hrheight = hrheight
        # self.image_loader = image_loader()
        trainset_dir = hr_dir
        hrdir = "hr7"
        find_txt = f"{trainset_dir}/{hrdir}/find.txt"
        if os.path.exists(find_txt):
            print(f"IMDBDataset loading {find_txt}")
            with open(find_txt, 'r') as f:
                self.hrFrames = np.asarray(
                    sorted(list(map(lambda line: os.path.join(trainset_dir, hrdir, line.strip()), f.readlines()))))
        else:
            print(f"IMDBDataset loading {trainset_dir}/{hrdir}/*/*.png")
            self.hrFrames = np.asarray(sorted(glob.glob(f'{trainset_dir}/{hrdir}/*/*.png')))

    def __getitem__(self, index):
        path = self.hrFrames[index]
        hr = read_sept(path)[1:6]
        assert len(hr) == 5
        H, W, C = hr[0].shape
        if W > self.hrheight:
            hr = [cv2.resize(img, (self.hrheight, self.hrheight), interpolation=cv2.INTER_LANCZOS4) for img in hr]
        H, W, C = hr[0].shape
        assert H == W and W == self.hrheight, f"hr WxH: {W}x{H}"
        lr = [cv2.resize(img, (W // 4, H // 4), interpolation=cv2.INTER_LINEAR) for img in hr]
        h, w, c = lr[0].shape
        assert h == w and w == (self.hrheight // 4), f"lr WxH: {w}x{h}"

        tlr = torch.tensor(np.array(lr)).transpose(1, 3) / 255.
        thr = torch.tensor(np.array(hr)).transpose(1, 3) / 255.
        return tlr, thr

    def __len__(self):
        return len(self.hrFrames)


class FRDataset(data.Dataset):

    def __init__(self, hr_dir, upscale_factor):
        self.file_hr_dir = hr_dir
        self.transform = base_transform
        self.upscale_factor = upscale_factor
        # self.image_loader = image_loader()
        self.hr_frames_set = os.listdir(self.file_hr_dir)

    def __getitem__(self, index):
        def get_from_set(dir, frame_set):
            frames = frame_set[index]  # 0266
            # print(f'frame is {frames}, typ is {type(frames)}')
            # frame_tensor = torch.Tensor(size=(frame_counter, 3, self.height, self.weight))
            frame_tensor = []

            # file_dir_frames = self.file_dir + frames
            file_dir_frames = os.path.join(dir, frames)
            imgs_path = os.listdir(file_dir_frames)
            imgs_path.sort()

            i = 0
            for img in imgs_path:
                final_path = file_dir_frames + "/" + img
                # final_path = '/'.os.listdir(file_dir_frames,img)
                img_tensor = image_loader(final_path)
                # print(img_tensor.size())
                frame_tensor.append(img_tensor)
                i = i + 1
            res = torch.stack(frame_tensor, dim=0)
            # print(f'res has shape {res.shape}')
            return res

        def get_lr_from_set(dir, frame_set, upscale_factor):
            frames = frame_set[index]  # 0266
            # print(f'frame is {frames}, typ is {type(frames)}')
            # frame_tensor = torch.Tensor(size=(frame_counter, 3, self.height, self.weight))
            frame_tensor = []

            # file_dir_frames = self.file_dir + frames
            file_dir_frames = os.path.join(dir, frames)
            imgs_path = os.listdir(file_dir_frames)
            imgs_path.sort()
            i = 0
            for img in imgs_path:
                final_path = file_dir_frames + "/" + img
                # final_path = '/'.os.listdir(file_dir_frames,img)
                img_tensor = lr_image_loader(final_path, upscale_factor)
                # print(img_tensor.size())
                frame_tensor.append(img_tensor)
                i = i + 1
            res = torch.stack(frame_tensor, dim=0)
            # print(f'res has shape {res.shape}')
            return res

        return get_lr_from_set(self.file_hr_dir, self.hr_frames_set, self.upscale_factor), \
               get_from_set(self.file_hr_dir, self.hr_frames_set)

    def __len__(self):
        return len(self.hr_frames_set)

    # # this returns the basic infomation of the dataset.
    # def touch(self):


class loader_wrapper(object):
    def __init__(self, loader):
        self.loader = loader

    def __iter__(self):
        for lr_img, hr_img in self.loader:
            yield lr_img.permute(1, 0, 2, 3, 4), hr_img.permute(1, 0, 2, 3, 4)

    def __len__(self):
        return len(self.loader)


def get_data_loaders(batch, shuffle_dataset=True, dataset_size=0, validation_split=0.2, fixedIndices=-1):
    # batch = 4 # batch size of the data every time for training
    # batch_number = 100000  # number of batches, so we totally have batch_number * batch images
    # HR_height = height
    # HR_width = width
    #
    # LR_height = HR_height // SRFactor
    # LR_width = HR_width // SRFactor

    train_dir_HR = 'Data/HR'

    # FRData = FRDataset(hr_dir=train_dir_HR, upscale_factor=upscale_factor)
    imdb1 = IMDBDataset(hr_dir="/storage/datasets/imdb250_v2/train/fast_hd", upscale_factor=4, hrheight=256)
    imdb2 = IMDBDataset(hr_dir="/storage/datasets/imdb250_v2/validate/fast_hd", upscale_factor=4, hrheight=256)
    FRData = ConcatDataset([imdb1, imdb2])

    # data_loader_LR = data.DataLoader(FRData_LR, batch_size = batch, shuffle = True)
    # data_loader_HR = data.DataLoader(FRData_HR, batch_size = batch, shuffle = True)

    # print(data_loader[0].size())
    random_seed = 42
    if dataset_size == 0:
        dataset_size = len(FRData)
    print("Dataset size:", len(FRData))
    indices = list(range(dataset_size))
    split = int(np.floor(validation_split * dataset_size))
    if shuffle_dataset:
        np.random.seed(random_seed)
        np.random.shuffle(indices)
    indices = [fixedIndices] if fixedIndices != -1 else indices
    train_indices, val_indices = indices[split:], indices[:split]
    print("Training/Validation split: %s/%s", (1 - validation_split) * 100, validation_split * 100)
    print("Training samples chosen:", train_indices)
    print("Validation samples chosen:", val_indices)
    train_sampler = SubsetRandomSampler(train_indices)
    print("Training set size:", len(train_sampler))
    valid_sampler = SubsetRandomSampler(val_indices)
    print("Validation set size:", len(valid_sampler))

    train_loader = torch.utils.data.DataLoader(FRData, batch_size=batch, sampler=train_sampler, drop_last=True)
    validation_loader = torch.utils.data.DataLoader(FRData, batch_size=batch, sampler=valid_sampler, drop_last=True)
    train_loader = loader_wrapper(train_loader)
    validation_loader = loader_wrapper(validation_loader)

    return train_loader, validation_loader


if __name__ == "__main__":
    p = Image.open(
        "/storage/datasets/imdb250_v2/train/fast_128/hr7/./12.Years.a.Slave.2013.1080p.BluRay.2xRus.Eng..HQCLUB.mp4/scene100_1280_0.png")
    img_tensor = base_transform(p)

    # train, val = get_data_loaders(4)
    # for lr_img, hr_img in train:
    #     print(f'lr_img shape is {lr_img.shape}, hr_img shape is {hr_img.shape}')
    #     break
    imdb = IMDBDataset(hr_dir="/storage/datasets/imdb250_v2/train/fast_128", upscale_factor=4, hrheight=512)
    print(imdb[0])

# class TestFRVSR(unittest.TestCase):
#     def TestGetDataLoader(self):
#


# for i_batch, sample_batched in enumerate(zip(train_loader_LR, train_loader_HR)):
#        #print(sample_batched)
#        #print(data_loader_HR[i_batch].size())
#        permuted_LR_data = sample_batched[0].permute(1, 0, 2, 3, 4)
#        permuted_HR_data = sample_batched[1].permute(1, 0, 2, 3, 4) #labels
#        #print(permuted_data.contiguous())
#        print("LR:",permuted_LR_data.size())
#        print("HR:",permuted_HR_data.size())
#
# for j_batch, sample_batched in enumerate(zip(validation_loader_LR, validation_loader_HR)):
#        #print(sample_batched)
#        #print(data_loader_HR[i_batch].size())
#        permuted_LR_data = sample_batched[0].permute(1, 0, 2, 3, 4)
#        permuted_HR_data = sample_batched[1].permute(1, 0, 2, 3, 4) #labels
#        #print(permuted_data.contiguous())
#        print("LR:",permuted_LR_data.size())
#        print("HR:",permuted_HR_data.size())
