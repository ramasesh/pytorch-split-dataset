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

