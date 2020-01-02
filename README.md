# Split datasets in PyTorch

A commonly-studied continual learning scenario is using split datasets, which are subsets of a particular dataset which contain only a subset of labels.  For example, we can take splits of MNIST, say all 0s and 6s, or all 3s, 4s, and 7s, etc.  

Usage (for MNIST example):
```
import torchvision.datasets
import split_datasets

MNIST = torchvision.datasets.MNIST(root='dataset', train=True, download=False)
split_MNIST = split_datasets.SplitDataset(full_dataset=MNIST,
                                          split=[0,4],
                                          label_locations=locations)
```

Notes:
* label_locations is a dictionary in which keys are the labels of the original dataset; corresponding values are arrays with all of the item indices of the original dataset corresponding to that label.
* the order of items in the split dataset is the same as it was in the original dataset 

For convenience, the dictionaries for some common datasets (CIFAR10, CIFAR100, MNIST) are provided in the file `dataset_indices.json`.  Theses can be loaded as follows:

```
import json 

with open('dataset_indices.json', 'r') as f:
  data = json.load(f)

label_locations = data[DATASET_NAME][TRAIN or TEST]
```

(JSON rather than YAML b/c loading YAML turns out to be [way slower](https://stackoverflow.com/questions/27743711/can-i-speedup-yaml) in python; the annoyance of JSON is that dictionary keys are all strings)

The split dataset object has all the items in it, so the split can be changed without creating a new object:
```
split_MNIST.update_split([3,8])
```

