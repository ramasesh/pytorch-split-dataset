### record_labels.py
# records the indices of the labels in a given dataset and stores it in the file
# requires the dataset to already be downloaded
# arguments
#   dataset_name - which dataset to use
#   record_file - the YAML file which will store the labels
#   datafolder - folder in which the data lives

from absl import flags
from absl import app

FLAGS = flags.FLAGS

from torchvision import datasets
import yaml
import os

supported_datasets = {'MNIST': datasets.MNIST,
                      'CIFAR10': datasets.CIFAR10, 
                      'CIFAR100': datasets.CIFAR100}

flags.DEFINE_enum('dataset_name', None, list(supported_datasets.keys()), 'dataset to record')
flags.DEFINE_string('datafolder', None, 'directory containing data')
flags.DEFINE_string('record_file', None, 'file containing data record')

def main(argv):
    dataset_train = supported_datasets[FLAGS.dataset_name](root=FLAGS.datafolder,
                                                           train=True,
                                                           download=True)

    dataset_test = supported_datasets[FLAGS.dataset_name](root=FLAGS.datafolder,
                                                           train=False,
                                                           download=True)

    num_classes = len(dataset_train.classes)
   
    train_label_locs = {i: [] for i in range(num_classes)}
    test_label_locs = {i: [] for i in range(num_classes)}

    for loc, img in enumerate(dataset_train):
        train_label_locs[img[1]].append(loc) 
    for loc, img in enumerate(dataset_test):
        test_label_locs[img[1]].append(loc) 

    all_label_locs = {FLAGS.dataset_name: {'train': train_label_locs, 
                                           'test': test_label_locs}}

    if os.path.exists(FLAGS.record_file):
        with open(FLAGS.record_file, 'r') as f:
            current_data = yaml.safe_load(f)
        current_data.update(all_label_locs)
    else:
        current_data = all_label_locs

    with open(FLAGS.record_file, 'w') as f:
        yaml.safe_dump(current_data, f)

if __name__=='__main__':
    app.run(main)
