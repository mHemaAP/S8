import torch
import torchvision

torch.manual_seed(11)

### Download the train data and store in folder 'data' in the current folder
### Transformations are to be applied on this train data
def download_train_data(train_transforms):
    return torchvision.datasets.CIFAR10('./data', 
                                        train=True, 
                                        download=True, 
                                        transform=train_transforms)

### Download the test data and store in folder 'data' in the current folder
### Transformations are to be applied on this train data
def download_test_data(test_transforms):
    return torchvision.datasets.CIFAR10('./data', 
                                        train=False, 
                                        download=True, 
                                        transform=test_transforms)

### Load the train and test data as to be used by the model
### Batch Size is set to be 512 if the GPU is available 
def get_loader(train_dataset, test_dataset, use_cuda=True):
    # dataloader arguments 
    dataloader_args = dict(shuffle=True, 
                       batch_size=512, 
                       num_workers=4, 
                       pin_memory=True) if use_cuda else dict(shuffle=True, 
                                                              batch_size=64)
    # train dataloader
    train_loader = torch.utils.data.DataLoader(train_dataset, 
                                               **dataloader_args)
    # test dataloader
    test_loader = torch.utils.data.DataLoader(test_dataset, 
                                              **dataloader_args)
    
    return train_loader, test_loader