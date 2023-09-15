import os
from os import sep
import torch

from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

class ImageDataset(Dataset):
    def __init__(self, input_dir: str, target_dir: str, transform=None):
        def _get_number_of_files_in_dir(root_dir: str) -> int:
            """Count number of files in dir and subdirs.

            NOTE: This is incredibly slow. Consider structuring data dirs differently, so rather than having one input dir, and a separate target dir, 
            input and target img is in the same dir: Something like: dataset/img1/target.png, dataset/img1/input.png.
            It does not make this function faster, but it may remove the need for it, since length of input and target does not need
            to be validated; the exception can be raised during __getitem__ if the dir does not contain both target and input.

            Args:
                root_dir (str): The root of the dir, which files are to be counted

            Returns:
                int: The number of (non dir) files in the dir and subdirs
            """
            return sum([len(files) for _root, _dirs, files in os.walk(root_dir)])
        
        self.input_dir = input_dir
        self.target_dir = target_dir
        self.transform = transform

        # validate length of inputs before setting self._len
        input_len = _get_number_of_files_in_dir(input_dir)
        target_len = _get_number_of_files_in_dir(target_dir)

        if input_len != target_len:
            raise Exception("Invalid input. Number of input samples does not equal number of target samples.")
        
        self._len = input_len

    def __len__(self) -> int:
        """Returns the length of the dataset. 

        Returns:
            int: The amount of samples in the dataset
        """
        return self._len
    
    
    

def test_dataset():
    dataset: ImageDataset = ImageDataset(f"dataset{sep}input_bw", f"dataset{sep}target", None)

    print(dataset.__len__())

def load_image_data():
    def crop_left(image):
        return transforms.functional.crop(image, 0, 0, 1600, 1040)
    
    transform = transforms.Compose([transforms.Lambda(crop_left),
                                    transforms.Resize(1040),
                                    transforms.ToTensor()])
    dataset = datasets.ImageFolder(f"dataset", transform=transform)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=57, shuffle=True)

    imgs, labs = next(iter(dataloader))
    img = imgs[2]
    img = img.permute(1,2,0)
    plt.imshow(img)
    plt.show()


test_dataset()