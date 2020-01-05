import numpy as np
from torchvision import datasets
import json
    #json was chosen because it's faster tham YAML, but has the downside of requiring keys to be string
class SplitCIFAR10(datasets.CIFAR10):

    def __init__(self, root, train, download, split, config_file, transform=None, target_transform=None):
        # config_file gives the locations of the images by label in the original dataset 
        super(SplitCIFAR10, self).__init__(root=root,
                                               train=train,
                                               download=download,
                                               transform=transform,
                                               target_transform=target_transform)
        
        with open(config_file, 'r') as f:
            label_location_dict = json.load(f)
        if train:
            self.label_locations = label_location_dict['CIFAR10']['train']
        else:
            self.label_locations = label_location_dict['CIFAR10']['test']
       
        self.update_split(split)

    def __len__(self):
        return len(self.indices_in_split)

    def __getitem__(self, idx):
        return super(SplitCIFAR10, self).__getitem__(self.indices_in_split[idx])

    def update_split(self, split):
        self.split = split
        self.indices_in_split = [self.label_locations[str(s)] for s in self.split] 
            #json requires dict keys to be string
        self.indices_in_split = sum(self.indices_in_split, [])
        self.indices_in_split = sorted(self.indices_in_split)

class SplitCIFAR100(datasets.CIFAR100):

    def __init__(self, root, train, download, split, config_file, transform=None, target_transform=None):
        # config_file gives the locations of the images by label in the original dataset 
        super(SplitCIFAR100, self).__init__(root=root,
                                               train=train,
                                               download=download,
                                               transform=transform,
                                               target_transform=target_transform)
        
        with open(config_file, 'r') as f:
            label_location_dict = json.load(f)
        if train:
            self.label_locations = label_location_dict['CIFAR100']['train']
        else:
            self.label_locations = label_location_dict['CIFAR100']['test']
       
        self.update_split(split)

    def __len__(self):
        return len(self.indices_in_split)

    def __getitem__(self, idx):
        return super(SplitCIFAR100, self).__getitem__(self.indices_in_split[idx])

    def update_split(self, split):
        self.split = split
        self.indices_in_split = [self.label_locations[str(s)] for s in self.split] 
            #json requires dict keys to be string
        self.indices_in_split = sum(self.indices_in_split, [])
        self.indices_in_split = sorted(self.indices_in_split)

class SplitMNIST(datasets.MNIST):

    def __init__(self, root, train, download, split, config_file, transform=None, target_transform=None):
        # config_file gives the locations of the images by label in the original dataset 
        super(SplitMNIST, self).__init__(root=root,
                                               train=train,
                                               download=download,
                                               transform=transform,
                                               target_transform=target_transform)
        
        with open(config_file, 'r') as f:
            label_location_dict = json.load(f)
        if train:
            self.label_locations = label_location_dict['MNIST']['train']
        else:
            self.label_locations = label_location_dict['MNIST']['test']
       
        self.update_split(split)

    def __len__(self):
        return len(self.indices_in_split)

    def __getitem__(self, idx):
        return super(SplitMNIST, self).__getitem__(self.indices_in_split[idx])

    def update_split(self, split):
        self.split = split
        self.indices_in_split = [self.label_locations[str(s)] for s in self.split] 
            #json requires dict keys to be string
        self.indices_in_split = sum(self.indices_in_split, [])
        self.indices_in_split = sorted(self.indices_in_split)

class SplitDataset():

    def __init__(self, full_dataset, split, label_locations):
        self.full_dataset = full_dataset
        self.label_locations = label_locations
        self.update_split(split)

    def __len__(self):
        return len(self.indices_in_split)

    def __getitem__(self, idx):
        return self.full_dataset.__getitem__(self.indices_in_split[idx])

    def update_split(self, split):
        self.split = split
        self.indices_in_split = [self.label_locations[str(s)] for s in self.split] 
            #json requires dict keys to be string
        self.indices_in_split = sum(self.indices_in_split, [])
        self.indices_in_split = sorted(self.indices_in_split)

class DownsampledDataset():
    # Dataset with a subset of images from the original labeled dataset,
    #   with an equal number of 
    def __init__(self, full_dataset, num_pts_per_class, label_locations, random_seed=0):
        # seed is for selectin the elements of each 
        self.full_dataset = full_dataset
        self.label_locations = label_locations
      
        self.__set_selected_indices__(random_seed, num_pts_per_class)

    def __set_selected_indices__(self, random_seed, num_pts_per_class):
        np.random.seed(random_seed)
        self.selected_indices = []
        for key in self.label_locations.keys():
            self.selected_indices.append(np.random.choice(self.label_locations[key], size=num_pts_per_class))
        self.selected_indices = np.array(self.selected_indices, dtype=int)   
        self.selected_indices = self.selected_indices.flatten('F') # column-major flattening
         
    def __len__(self):
        return len(self.selected_indices)
    
    def __getitem__(self, idx):
        return self.full_dataset.__getitem__(self.selected_indices[idx])
