import time
import torch
from torchvision import datasets, transforms, models

# Function to load and preprocess the data
def load_data(data_dir):
    # Training data augmentation: random scaling, rotations, mirroring, and/or cropping
    # Data normalization: The training, validation, and testing data is appropriately cropped and normalized
    # For the means, it's [0.485, 0.456, 0.406] and for the standard deviations [0.229, 0.224, 0.225]
    # These values will shift each color channel to be centered at 0 and range from -1 to 1

    # Training dataset is randomly scaled, transformed, and flipped
    train_transforms = transforms.Compose([transforms.RandomRotation(50),
                                           transforms.RandomResizedCrop(224),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406],
                                                                [0.229, 0.224, 0.225])])

    # Validation and testing datasets share the same transformations with no scaling or rotation
    valid_test_transforms = transforms.Compose([transforms.Resize(256),
                                                transforms.CenterCrop(224),
                                                transforms.ToTensor(),
                                                transforms.Normalize([0.485, 0.456, 0.406],
                                                                     [0.229, 0.224, 0.225])])

    # Data loading: The data for each set (train, validation, test) is loaded with torchvision's ImageFolder
    train_data = datasets.ImageFolder(data_dir + '/train', transform=train_transforms)
    valid_data = datasets.ImageFolder(data_dir + '/valid', transform=valid_test_transforms)
    test_data = datasets.ImageFolder(data_dir + '/test', transform=valid_test_transforms)

    # Data batching: The data for each set is loaded with torchvision's DataLoader
    # Training data is shuffled to prevent unintended bias
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(valid_data, batch_size=64)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=64)

    return train_loader, valid_loader, test_loader, train_data, test_data, valid_data


# Define validation function for the model
def validation(model, valid_loader, criterion, gpu_mode):
    accuracy = 0
    valid_loss = 0

    if gpu_mode == True:
        model.to('cuda')
    else:
        pass

    # Iterate over images and labels from valid_loader dataset
    for images, labels in valid_loader:

        if gpu_mode == True:
            images, labels = images.to('cuda'), labels.to('cuda')
        else:
            pass

        # Perform forward pass
        output = model.forward(images)

        # Calculate validation loss
        valid_loss += criterion(output, labels).item()

        # Return a new tensor with the exponential of the elements of the input tensor to calculate probability
        ps = torch.exp(output)

        # Calculate accuracy based on the maximum probability versus actual flower label
        match = (labels.data == ps.max(dim=1)[1])
        accuracy += match.type(torch.FloatTensor).mean()

    return valid_loss, accuracy

# Training the network: The parameters of the feedforward classifier are appropriately trained
def train_model(model, epochs, train_loader, valid_loader, criterion, optimizer, gpu_mode):
    # An iteration = 1 forward and 1 backward pass
    iteration = 0
    print_line = 10

    if gpu_mode == True:
        model.to('cuda')
    else:
        pass

    for epoch in range(epochs):
        running_loss = 0
        since = time.time()

        # Iterate over images and labels from train_loader and valid_loader dataset
        for inputs, labels in train_loader:

            iteration += 1

            if gpu_mode == True:
                inputs, labels = inputs.to('cuda'), labels.to('cuda')
            else:
                pass

            # Reset gradient back to 0 because here we perform mini-batch gradient descent
            optimizer.zero_grad()

            # Perform forward pass and compute the loss
            outputs = model.forward(inputs)
            loss = criterion(outputs, labels)

            # Perform backward pass to compute the gradients with respect to model parameters
            loss.backward()

            # Update parameters in the direction of the gradients
            optimizer.step()

            # Calculate the training loss
            running_loss += loss.item()

            # Validate model
            if iteration % print_line == 0:
                # Set model to evaluation mode
                model.eval()

                # Turn off gradients to reduce memory usage as it is not needed in validation mode
                with torch.no_grad():
                    valid_loss, accuracy = validation(model, valid_loader, criterion, gpu_mode)

                # Validation Loss and Accuracy: During training, the validation loss and accuracy are displayed

                print(f"Epoch {epoch+1}, \
				Training Loss: {round(running_loss/print_line,2)} \
				Validation Loss: {round(valid_loss/len(valid_loader),2)} \
				Validation Accuracy: {round(float(accuracy/len(valid_loader)),2)}")

                # Reset training loss
                running_loss = 0

                # Set model back to training mode
                model.train()

        # Calculate and print time elapsed in each epoch
        time_elapsed = time.time() - since
        print('Epoch duration: {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    return model, optimizer


# Testing Accuracy: The network's accuracy is measured on the test data
def test_model(model, test_loader, gpu_mode):
    accurate = 0
    total = 0

    if gpu_mode == True:
        model.to('cuda')
    else:
        pass

    # Turn off gradients to reduce memory usage as it is not needed in testing mode
    with torch.no_grad():

        for data in test_loader:
            images, labels = data

            # Convert inputs and labels to cuda tensors
            if gpu_mode == True:
                images, labels = images.to('cuda'), labels.to('cuda')
            else:
                pass

            # Obtain model output values from images
            outputs = model(images)

            # Designate maximum probability output as predicted flower class
            _, predicted = torch.max(outputs.data, 1)

            # Add total number of images to total image counter
            total += labels.size(0)

            # Add accurately classified images to accurate image counter
            accurate += (predicted == labels).sum().item()

    # Print neural network accuracy calculated from accurate/total image counters
    print(f"Neural network accuracy: {round(100 * accurate / total,2)}%")


