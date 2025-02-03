import matplotlib.pyplot as plt
import numpy as np

# Function to unnormalize and show an image
def imshow(img):
    '''
    Displays a single CIFAR-10 image after unnormalizing it.
    
    Args:
    - img (torch.Tensor): A normalized image tensor with shape (3, 32, 32) representing CIFAR-10 image.
    
    Functionality:
    - Unnormalizes the image (brings pixel values from [-1, 1] range to [0, 1] range).
    - Converts the image tensor to a NumPy array.
    - Displays the image using matplotlib.
    '''
    img = img / 2 + 0.5  # Unnormalize the image
    npimg = img.cpu().numpy().reshape(3, 32, 32)  # Convert to NumPy array and reshape to (3, 32, 32)
    plt.imshow(np.transpose(npimg, (1, 2, 0)))  # Transpose to match image format (height, width, channels)
    plt.show()

# Function to visualize predictions for CIFAR-10
def visualize_predictions(images, predicted, actual, idx, classes, num_images=5):
    """
    Visualize model predictions on CIFAR-10 dataset.

    Args:
    - images (torch.Tensor): A batch of image tensors from the test dataset.
    - predicted (torch.Tensor): The predicted labels from the model for each image in the batch.
    - actual (torch.Tensor): The actual labels of the images.
    - idx (int): The figure index for saving the visualized predictions.
    - classes (list of str): The class names in CIFAR-10, used to map labels to human-readable names.
    - num_images (int): Number of images to visualize (default is 5).
    
    Functionality:
    - Loops through a given number of images (num_images) in the batch.
    - For each image, it displays the actual and predicted class labels.
    - The function saves the visualized predictions as a JPG image in the 'figures' folder with a filename `prediction_{idx}.jpg`.
    - Displays the images in a matplotlib figure.
    """
    plt.figure(figsize=(10, 6))
    
    for img_index in range(num_images):
        image = images[img_index]
        actual_label = classes[actual[img_index]]  # Get actual class name
        predicted_label = classes[predicted[img_index]]  # Get predicted class name
        
        plt.subplot(1, num_images, img_index + 1)  # Plot each image in a subplot
        imshow(image)
        plt.title(f"Predicted: {predicted_label}\nActual: {actual_label}")  # Show actual vs predicted
        plt.axis('off')  # Hide axis for better display
    
    # Save the figure to the 'figures' folder
    plt.savefig(f'figures/prediction_{idx}.jpg')
    plt.tight_layout()
    plt.show()

# Function to visualize training and validation loss
def visualize_train_val_losses(train_losses, val_losses):
    """
    Plot and visualize the training and validation loss over epochs.

    Args:
    - train_losses (list of floats): A list containing the training loss for each epoch.
    - val_losses (list of floats): A list containing the validation loss for each epoch.

    Functionality:
    - Plots the training and validation losses on the same graph.
    - Adds labels for the x-axis (Epochs) and y-axis (Loss).
    - Adds a title and legend to the graph for clarity.
    - Saves the plot as a JPG image in the 'figures' folder with the filename 'train_val_loss.jpg'.
    - Displays the loss curves using matplotlib.
    """
    plt.figure(figsize=(10, 5))
    
    # Plot training loss
    # Plot validation loss
    
    # Label for x-axis
    # Label for y-axis
    # Title for the plot
    # Display legend
    
    # Save the figure to the 'figures' folder

    
    # Show the plot

