from torchvision import datasets
import yaml

class SplitCIFAR10(datasets.CIFAR10):

    def __init__(self, root, train, download, split, config_file, transform=None, target_transform=None):
        # config_file gives the locations of the images by label in the original dataset 
        super(SplitCIFAR10, self).__init__(root=root,
                                               train=train,
                                               download=download,
                                               transform=transform,
                                               target_transform=target_transform)
        
        print("Super initted")
        with open(config_file, 'r') as f:
            label_location_dict = yaml.load(f)
        if train:
            self.label_locations = label_location_dict['CIFAR10']['train']
        else:
            self.label_locations = label_location_dict['CIFAR10']['test']
        print("YAML loaded")
       
        self.update_split(split)

    def __len__(self):
        return len(self.indices_in_split)

    def __getitem__(self, idx):
        return super(SplitCIFAR10, self).__getitem__(self.indices_in_split[idx])

    def update_split(self, split):
        self.split = split
        self.indices_in_split = [self.label_locations[s] for s in self.split] 
        self.indices_in_split = sum(self.indices_in_split, [])
        self.indices_in_split = sorted(self.indices_in_split)

