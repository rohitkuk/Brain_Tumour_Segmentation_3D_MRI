import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision import datasets


class Brats2020Dataset2020(Dataset):

    URL = 'https://drive.google.com/u/0/uc?id=122AF3SFPhp3wUu4QBnoQMJR2BzPmWcdq&export=download'
    URL = 'https://drive.google.com/uc?id=0B9P1L--7Wd2vNm9zMTJWOGxobkU'
    OUT_FILE = 'miccai_brats.zip'
    UNZIP_FOLDER = 'dataset/miccai_brats'

    def __init__(self, root, train=True, transform=None, target_transform=None, download=False):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform
        self.train = train  # training set or test set

        if download and not self._check_exists():
            self.download()
            self.extract()

        if not self._check_exists():
            raise RuntimeError('Dataset not found.' +
                               ' You can use download=True to download it')

        if self.train:
            self.train_data, self.train_labels = torch.load(
                os.path.join(self.root, self.processed_folder, self.training_file))
        else:
            self.test_data, self.test_labels = torch.load(
                os.path.join(self.root, self.processed_folder, self.test_file))

    def __len__(self):
        if self.train:
            return len(self.train_data)
        else:
            return len(self.test_data)

    def _check_exists(self):
        return os.path.exists(os.path.join(self.root, UNZIP_FOLDER))

    def download(self):
        # Downloading the Dataset
        import gdown
        print("Starting Dwonload !! ")
        gdown.download(URL, OUT_FILE, quiet=False, )
        print('Done!')

    def extract(self):

        from zipfile import ZipFile
        from tqdm import tqdm
        print("Unzipping the File")

        with ZipFile(file=file_name) as zip_file:
            for file in tqdm(iterable=zip_file.namelist(), total=len(zip_file.namelist())):
                zip_file.extract(member=OUT_FILE, path=UNZIP_FOLDER)

        print("Done")

    def arrange(self):
        from glob import glob
        # Removing the Zipped File
        print("Removing the Zipped File")
        os.remove(OUT_FILE)
        print("Removing the unwated files")

        folder_prefix = "BraTS20_Training"
        all_files = glob(os.path.join(self.root + UNZIP_FOLDER )+ "/{instance_folder}*/{instance_folder}*.nii.gz")

        for i in range(all_files):
            if not i.endswith('t1ce.nii.gz') and not i.endswith('seg.nii.gz'):
                os.remove(i)

   

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        # """
        # if self.train:
        #     img, target = self.train_data[index], self.train_labels[index]
        # else:
        #     img, target = self.test_data[index], self.test_labels[index]

        # img = wrapper(img)
        # img = Image.fromarray(img, mode='L')

        # if self.transform:  # check for not None
        #     img = self.transform(img)

        # if self.target_transform:
        #     target = self.target_transform(target)
        img = None
        target = None

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
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(
            tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str
