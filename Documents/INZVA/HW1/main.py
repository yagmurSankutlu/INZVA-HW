import torch
import torch.nn as nn
from utils.model import NeuralNet
from utils.data_utils import *
from utils.viz_utils import *
from training import train_model
from testing import test_model

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
    epochs = # Number of epochs for training
    batch_size = # Number of samples per batch
    learning_rate = # Learning rate for the optimizer

    # Define the device for computation (GPU if available, otherwise CPU)
    device = #

    # Initialize the model, loss function, and optimizer
    model = # Initiliaze and move the model to the device
    criterion = # Define the loss function (CrossEntropy for classification)
    optimizer = # Choose optimizer with a learning rate



    # Get the training and test datasets
    train_set, test_set = #
    
    # Create DataLoader for the training set
    train_loader = #
    
    # Split the test set into validation and test subsets
    val_indices, test_indices = #
    
    # Create DataLoaders for validation and test subsets
    val_loader  = # 
    test_loader = # 
    
    # Train the model and track losses for each epoch
    train_losses, val_losses = #
    
    # Visualize the training and validation loss curves
    #

    # Test the model on the test dataset, get visualizations and print accuracy
    #

if __name__ == '__main__':
    main()
