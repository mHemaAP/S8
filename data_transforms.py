# Returns a list of transformations when called
import torch
import torchvision

torch.manual_seed(11)

### Return the train data after the transformations are applied
def get_train_transforms():
    train_transformations = [ # Applies transformations on the image so it can be perfect for our model.
#             transforms.TrivialAugmentWide(interpolation=transforms.InterpolationMode.BILINEAR),
#             transforms.RandomHorizontalFlip(),


        # transforms.RandomCrop(32, padding=4),
        # transforms.RandomHorizontalFlip(), # Flips the image w.r.t horizontal axis
        # transforms.RandomRotation((-7,7)), # Rotates the image to a specified angel
        # transforms.RandomAffine(0, shear=10, scale=(0.8,1.2)), # Performs actions like zooms, 
        # change shear angles.

        torchvision.transforms.ColorJitter(brightness=0, 
                                           contrast=0.1, 
                                           saturation=0.2), # Set the color params


        torchvision.transforms.ToTensor(), # comvert the image to tensor so that it can work with torch
        torchvision.transforms.Normalize((0.491, 0.482, 0.446), (0.247, 0.243, 0.261)) #Normalize all the images
        ]

    return train_transformations

### Return the train data after the transformations are applied
def get_test_transforms():
    test_transforms = [
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.491, 0.482, 0.446), (0.247, 0.243, 0.261))
    ]
    return test_transforms
