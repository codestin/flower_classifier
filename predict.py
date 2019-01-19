# Predicting classes: The predict.py script successfully reads in an image and a checkpoint then prints the most likely image class and it's associated probability
# Top K classes: The predict.py script allows users to print out the top K classes along with associated probabilities
# Displaying class names: The predict.py script allows users to load a JSON file that maps the class values to other category names
# Predicting with GPU: The predict.py script allows users to use the GPU to calculate the predictions

import argparse
import json
import numpy as np
import torch
from torchvision import datasets, transforms, models
from PIL import Image

parser = argparse.ArgumentParser(description='Take the path to an image and a checkpoint, then return the top K most probably classes for that image')

parser.add_argument('--image_dir', action='store',
					default = '../aipnd-project/flowers/test/2/image_05100',
					help='Set path to image (default = ../aipnd-project/flowers/test/2/image_05100)')

parser.add_argument('--save_dir', action='store', default = 'checkpoint.pth',
					help='Set load location for checkpoint (default = checkpoint.pth)')

parser.add_argument('--topk', action='store',
					dest='topk', type=int, default = 5,
					help='Set top K flower class (default = 5)')

parser.add_argument('--cat_to_name', action='store',
					dest='cat_name_dir', default = 'cat_to_name.json',
					help='Set path to mapping from category label to category name (default = cat_to_name.json')

parser.add_argument('--gpu', action="store_true", default=True,
					help='Set GPU usage (default = True)')

results = parser.parse_args()

image_dir = results.image_dir
topk = results.topk
cat_names = results.cat_name_dir
gpu_mode = results.gpu
save_dir = results.save_dir

# Loading checkpoints: There is a function that successfully loads a checkpoint and rebuilds the model

checkpoint = torch.load(save_dir)
if checkpoint['arch'] == 'vgg16':
    model = models.vgg16(pretrained=True)

elif checkpoint['arch'] == 'alexnet':
    model = models.alexnet(pretrained=True)

model.state_dict (checkpoint['state_dict'])
model.classifier = checkpoint['classifier']
model.class_to_idx = checkpoint['class_to_idx']

for param in model.parameters():
    param.requires_grad = False

# Load in dictionary mapping the integer encoded categories to the actual names of the flowers
with open('cat_to_name.json', 'r') as f:
	cat_to_name = json.load(f)

# Image Processing: Converts a PIL image into an object that can be used as input to a trained model
def process_image(image_dir):
    # Use PIL to load the image
    pil_image = Image.open(f'{image_dir}' + '.jpg')

    # Normalize image per network expectations:
    # For the mean, it's [0.485, 0.456, 0.406] and for the standard deviations [0.229, 0.224, 0.225]
    # Resize the images where the shortest side is 256 pixels, keeping the aspect ratio
    # Crop out the center 224x224 portion of the image
    transform = transforms.Compose([transforms.Resize(256),
                                    transforms.CenterCrop(224),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406],
                                                         [0.229, 0.224, 0.225])])

    pil_transform = transform(pil_image)

    # Convert PyTorch tensor to Numpy array
    np_image = np.array(pil_transform)

    # Convert from Numpy array to PyTorch tensor
    img_tensor = torch.from_numpy(np_image).type(torch.FloatTensor)

    # Add batch size argument
    processed_image = img_tensor.unsqueeze(0)

    return processed_image

# Class Prediction: The predict function returns the top K most probably classes for that image
def predict(image_dir, model, topk, gpu_mode):
    image = process_image(image_dir)

    # Switch to GPU mode
    if gpu_mode == True:
        model.to('cuda')
    else:
        model.cpu()

    # Convert processed image to CUDA tensor
    if gpu_mode == True:
        image = image.to('cuda')
    else:
        pass

    with torch.no_grad():
        model_output = model.forward(image)

    # Calculate and store top probabilities using torch.topk(k), which returns the highest probabilities and the indices
    probs = torch.exp(model_output)
    probs_topk = probs.topk(topk)[0]
    idx_topk = probs.topk(topk)[1]

    # Convert stored top probabilities to Numpy arrays
    probs_topk_array = np.array(probs_topk)[0]
    idx_topk_array = np.array(idx_topk)[0]

    # Convert index to class labels using class_to_idx
    class_to_idx = model.class_to_idx

    # Invert the dictionary to obtain mapping from index to class
    idx_to_class = {val: key for key, val in class_to_idx.items()}
    class_topk_array = []
    for idx in idx_topk_array:
        class_topk_array += [idx_to_class[idx]]

    return probs_topk_array, class_topk_array

# Predict the probabilities and classes
probs, classes = predict(image_dir, model, topk, gpu_mode)

# Convert from the class integer encoding to actual flower names with the cat_to_name.json
names = [cat_to_name[i] for i in classes]

# Print name of top K predicted flower class and probability
print(f"This is a '{names[0]}' with a probability of {round(probs[0]*100,2)}% ")