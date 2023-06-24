import numpy as np
import torch
import torchvision
import matplotlib.pyplot as plt


def GetCorrectPredCount(pPrediction, pLabels):
    return pPrediction.argmax(dim=1).eq(pLabels).sum().item()


test_incorrect_pred = {"images": [], "ground_truths": [], "predicted_vals": []}

#####  New Logic additions

def GetInCorrectPreds(pPrediction, pLabels):
    pPrediction = pPrediction.argmax(dim=1)
    indices = pPrediction.ne(pLabels).nonzero().reshape(-1).tolist()
    return indices, pPrediction[indices].tolist(), pLabels[indices].tolist()


def get_incorrect_test_predictions(model, device, test_loader):
    model.eval()

    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)

            output = model(data)

            ind, pred, truth = GetInCorrectPreds(output, target)
            test_incorrect_pred["images"] += data[ind]
            test_incorrect_pred["ground_truths"] += truth
            test_incorrect_pred["predicted_vals"] += pred

    return test_incorrect_pred


def display_train_data(train_data):

  print('[Train]')
  print(' - Numpy Shape:', train_data.cpu().numpy().shape)
  print(' - Tensor Shape:', train_data.size())
  print(' - min:', torch.min(train_data))
  print(' - max:', torch.max(train_data))
  print(' - mean:', torch.mean(train_data))
  print(' - std:', torch.std(train_data))
  print(' - var:', torch.var(train_data))

def show_image_by_index(images, index):  
  plt.imshow(images[index].numpy().squeeze(), cmap='gray_r')

def de_normalize():
    inv_normalize = torchvision.transforms.Normalize(
        mean=[-0.491/0.247, -0.482/0.243, -0.446/0.261],
        std=[1/0.247, 1/0.243, 1/0.261]
    )
    return inv_normalize

def show_cifar10_images(images, labels, classes):

  # for i in range(0, 10):
  #   print(repr(i) + " - " + repr(classes[i]))

  for i in range(25):
    plt.subplot(5, 5, i + 1)
    plt.tight_layout()  
    inv_norm = de_normalize()
    img_inv_tensor = inv_norm(images[i])


    npimg = img_inv_tensor.numpy()
    npimg = np.transpose(npimg, (1, 2, 0))
    plt.imshow(npimg)
    plt.title(repr(classes[labels[i]]))    

    # plt.title("Image " +
    #     repr(labels[i])
    #     + "- " 
    #     + repr(classes[labels[i]]))

    plt.xticks([])
    plt.yticks([])


def show_cifar10_incorrect_predictions(incorrect_prediction, classes):

  for i in range(0, 10):
    print(repr(i) + " - " + repr(classes[i]), end=" ")

  for i in range(16):
    plt.subplot(4, 4, i + 1)
    plt.tight_layout()

    tensor_incorrect_pred = incorrect_prediction["images"][i].cpu().squeeze(0)

    inv_norm = de_normalize()

    img_inv_tensor = inv_norm(tensor_incorrect_pred)

    npimg = img_inv_tensor.numpy()
    npimg = np.transpose(npimg, (1, 2, 0))
    plt.imshow(npimg)

    plt.title("Pred " +
        repr(incorrect_prediction["predicted_vals"][i])
        + " vs " + "Truth "
        + repr(incorrect_prediction["ground_truths"][i])
    )
    plt.xticks([])
    plt.yticks([])


def display_multiple_images(images, num_of_images):
    
    figure = plt.figure()
  
    for index in range(1, num_of_images + 1):
        plt.subplot(6, 10, index)
        plt.axis('off')
        plt.imshow(images[index].numpy().squeeze(), cmap='gray_r')
    
        
def display_model_stats(train_loss, train_accuracy, test_loss, test_accuracy):
  fig, axs = plt.subplots(2,2,figsize=(15,10))
  axs[0, 0].plot(train_loss)
  axs[0, 0].set_title("Training Loss")
  axs[1, 0].plot(train_accuracy)
  axs[1, 0].set_title("Training Accuracy")
  axs[0, 1].plot(test_loss)
  axs[0, 1].set_title("Test Loss")
  axs[1, 1].plot(test_accuracy)
  axs[1, 1].set_title("Test Accuracy")


def plot_test_incorrect_predictions(incorrect_pred):

    fig = plt.figure()
    for i in range(16):
        plt.subplot(4, 4, i + 1)
        plt.tight_layout()
        plt.imshow(incorrect_pred["images"][i].cpu().squeeze(0), cmap="gray")
        plt.title(
            repr(incorrect_pred["predicted_vals"][i])
            + " vs "
            + repr(incorrect_pred["ground_truths"][i])
        )
        plt.xticks([])
        plt.yticks([])
