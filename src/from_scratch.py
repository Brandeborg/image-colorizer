import os
from os import sep
from pyparsing import Any
import torch

from torchvision import datasets, transforms, io
from torch.utils.data import Dataset

import matplotlib.pyplot as plt

class ImageDataset(Dataset):
    def __init__(self, input_dir: str, target_dir: str, transform=None):
        def _get_number_of_files_in_dir(root_dir: str) -> int:
            """Count number of files in dir and subdirs.

            NOTE: This is incredibly slow. Consider structuring data dirs differently, so rather than having one input dir, and a separate target dir, 
            input and target img is in the same dir. Something like: 
            dataset/img1/target.png, dataset/img1/input.png as opposed to dataset/target/img1.png, dataset/input/img1.png
            It does not make this function faster, but it may remove the need for it, since length of input and target does not need
            to be validated; the exception can be raised during __getitem__ if the dir does not contain both target and input.

            Args:
                root_dir (str): The root of the dir, which files are to be counted

            Returns:
                int: The number of (non dir) files in the dir and subdirs
            """
            return sum([len(files) for _root, _dirs, files in os.walk(root_dir)])
        
        def _create_index2filename_map(root_dir: str) -> dict:
            """Create a mapping from index to file name, to be used in __getitem__. 
            File names include path from (not including) root_dir.

            Args:
                root_dir (str): dir in which to recursively look for filename

            Returns:
                dict: Map from index to file name
            """
            idx2file_map: dict = {}
            i = 0

            for root, _dirs, files in os.walk(root_dir):
                # extract dir leading up to file name, removing root_dir and first slash
                dir = root[len(root_dir)+1:]
                for file_name in files:
                    file_path = os.path.join(dir, file_name)
                    idx2file_map[i] = file_path
                    i += 1
            
            return idx2file_map
        
        self.input_dir = input_dir
        self.target_dir = target_dir
        self.transform = transform

        self._idx2file_map = _create_index2filename_map(input_dir)
        # validate length of target dir before setting self._len
        target_len = _get_number_of_files_in_dir(target_dir)

        if len(self._idx2file_map) != target_len:
            raise Exception("Invalid input. Number of target samples does not equal number of input samples.")
        
        self._len = target_len

    def __len__(self) -> int:
        """Returns the length of the dataset. 

        Returns:
            int: The amount of samples in the dataset
        """
        return self._len
    
    def __getitem__(self, index) -> Any:
        if torch.is_tensor(index):
            # really just "toscalar"
            index = index.tolist()

        input_path = os.path.join(self.input_dir, self._idx2file_map[index])
        target_path = os.path.join(self.target_dir, self._idx2file_map[index])

        input = io.read_image(input_path)
        target = io.read_image(target_path)

        if self.transform:
            # NOTE: this needs to be updated, if transform includes something with
            # randomization, as that would transform input and target differently
            # with current implementation
            input = self.transform(input)
            target = self.transform(target)

        return input, target
    
def test_dataset():
    def crop_left(image):
        return transforms.functional.crop(image, 0, 0, 1600, 1040)
    
    transform = transforms.Compose([transforms.Lambda(crop_left),
                                    transforms.Resize(1040, antialias=True)])
    dataset: ImageDataset = ImageDataset(f"dataset{sep}input_bw", f"dataset{sep}target", transform)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=57, shuffle=False)

    inputs, targets = next(iter(dataloader))
    input = targets[2]
    print(input)
    input = input.permute(1,2,0)
    #plt.imshow(input)
    #plt.show()


test_dataset()