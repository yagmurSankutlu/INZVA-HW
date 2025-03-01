import torch
import torch.nn as nn
from utils.model import NeuralNet
from utils.data_utils import *
from utils.viz_utils import *
from training import train_model
from testing import test_model
from torchvision import datasets, transforms

def main():
    '''
    The main function that coordinates the training and testing of the neural network on the CIFAR-10 dataset.

    Steps:
    - Sets the input size and number of classes for the CIFAR-10 dataset.
    - Defines the training parameters such as epochs, batch size, and learning rate.
    - Initializes the model, loss function, and optimizer.
    - Loads the training, validation, and test datasets.
    - Trains the model and visualizes the training and validation losses.
    - Tests the trained model on the test dataset and prints the accuracy.

    '''

    input_size = 3*32*32  # CIFAR-10 images are 3-channel RGB images with 32x32 pixels (3*32*32)
    num_classes = 10  # CIFAR-10 has 10 classes
    epochs = 5  # Number of epochs for training
    batch_size = 128  # Number of samples per batch
    learning_rate = 3e-4  # Learning rate for the optimizer
    # Define the device for computation (GPU if available, otherwise CPU)
    # Handle macOS graphic card by checking for Metal backend support
    if torch.backends.mps.is_available():
        device = torch.device('mps')
    elif torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    # Initialize the model, loss function, and optimizer
    model = NeuralNet(input_size, num_classes).to(device)
    criterion = nn.CrossEntropyLoss()  # Define the loss function (CrossEntropy for classification)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)  # Choose optimizer with a learning rate

    # Get the training and test datasets
    train_set, test_set = get_train_and_test_set()
    
    # Create DataLoader for the training set
    train_loader = get_trainloader(train_set, batch_size=batch_size)
    
    # Split the test set into validation and test subsets
    val_indices, test_indices = split_testset(test_set, 0.5)
    
    # Create DataLoaders for validation and test subsets
    val_loader  = get_validationloader(test_set, val_indices, batch_size)
    test_loader = get_testloader(test_set, test_indices, batch_size)
    
    # Train the model and track losses for each epoch
    train_losses, val_losses = train_model(model, device, epochs, criterion, optimizer, train_loader, val_loader)
    
    # Visualize the training and validation loss curves
    visualize_train_val_losses(train_losses, val_losses)
    
    # Test the model on the test dataset, get visualizations and print accuracy
    test_model(device, test_loader)


if __name__ == '__main__':
    main()
