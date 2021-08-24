import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision import datasets
import numpy as np

import os
from glob import glob

import gdown
import nibabel as nib
from zipfile import ZipFile
from tqdm import tqdm



class Brats2020Dataset2020(Dataset):

    URL = 'https://drive.google.com/uc?id=1fjhJKi6Cs71MpbTa_u4oHHKF3rO41F97&export=download'
    OUT_FILE = 'micca_train_2.zip'
    UNZIP_FOLDER = 'dataset/miccai_train'

    def __init__(self, root, train=True, transform=None, download=False):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.train = train  # training set or test set
        self.UNZIP_FOLDER = os.path.join(self.root, self.UNZIP_FOLDER)

        # Creating necessary Directories
        self.make_dirs()

        if download and not self._check_exists():
            self.download()
            self.extract()
            self.arrange()

        if not self._check_exists():
            raise RuntimeError('Dataset not found.' +
                               ' You can use download=True to download it')

        self.folder_prefix = "BraTS20_Training"

        self.all_files = glob(os.path.join(
            self.UNZIP_FOLDER) + "/{instance_folder}*/{instance_folder}*.gz".format(instance_folder=self.folder_prefix))
        
        self.images_t1c = np.array(
            sorted([file for file in self.all_files if file.endswith('t1ce.nii.gz')]))
        self.images_seg = np.array(
            sorted([file for file in self.all_files if file.endswith('seg.nii.gz')]))

        np.random.seed(42)
        self.perm = np.random.permutation(len(self.images_t1c))
        self.split = int(0.8 * len(self.perm))

        if self.train:
            self.images_t1c = self.images_t1c[self.perm[:self.split]]
            self.images_seg = self.images_seg[self.perm[:self.split]]
        else:
            self.images_t1c = self.images_t1c[self.perm[self.split:]]
            self.images_seg = self.images_seg[self.perm[self.split:]]

    def _check_exists(self):
        return os.path.exists(self.UNZIP_FOLDER)

    def make_dirs(self):
        dirslist = [self.UNZIP_FOLDER]
        for dir_ in dirslist:
            if not os.path.exists(dir_):
                os.mkdir

    def download(self):
        print("Dwonload Started !!!")
        gdown.download(self.URL, output=None, quiet=False)
        print("Dwonload Finished !!!")

    def extract(self):


        print("Unzipping the File")

        with ZipFile(file=os.path.join(self.root, self.OUT_FILE)) as zip_file:
            for file in tqdm(iterable=zip_file.namelist(), total=len(zip_file.namelist())):
                zip_file.extract(
                    member=file, path=os.path.join(self.root, 'dataset'))

        print("Done")

    def arrange(self):
        # Removing the Zipped File
        print("Removing the Zipped File")
        try:
            os.remove(os.path.join(self.root, self.OUT_FILE))
        except:
            pass
        print("Removing the unwated files")

        self.folder_prefix = "BraTS20_Training"

        self.all_files = glob(os.path.join(
            self.UNZIP_FOLDER) + "/{instance_folder}*/{instance_folder}*.gz".format(instance_folder=self.folder_prefix))

        for i in self.all_files:
            if not i.endswith('t1ce.nii.gz') and not i.endswith('seg.nii.gz'):
                os.remove(i)

    def __len__(self):
        return len(self.images_t1c)

    def __getitem__(self, index):

        img, target =  nib.load(self.images_t1c[index]), nib.load(self.images_seg[index])
        
        img, target = img.get_fdata(), target.get_fdata()

        target = ((target == 1) | (target == 4)).astype('float32')

        if self.transform:
            img = self.transform(img)
        return img, target


    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        tmp = 'train' if self.train is True else 'test'
        fmt_str += '    Split: {}\n'.format(tmp)
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(
            tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))

        return fmt_str
