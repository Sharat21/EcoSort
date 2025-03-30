from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.optim import Adam
import numpy as np
from json import loads, dump
from os import listdir
import re
from random import randint
import os

# Define labels and subclasses
labels = {
    'cardboard': 0,
    'metal': 1,
    'plastic': 2,
    'trash': 3,
    'glass': 4,
    'paper': 5
}

subclasses = {
    'plastic': ['PET', 'HDPE', 'PVC', 'LDPE', 'PP', 'PS'],
    'glass': ['clear_glass', 'brown_glass', 'green_glass'],
    'trash': ['organic', 'non_organic']
}

label_list = list(labels.keys())

# Define transformations
transform = transforms.Compose([
            transforms.Resize((300, 300)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) #Same normalization method used originally on efficient net -> Better for fine tuning
])


class ImageDataset(Dataset):
    def __init__(self, images, sub=False):

        self.images = images
        self.sub = sub
    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_path = list(self.images.keys())[idx]
        label = self.images[image_path]

        # Open the image
        if self.sub:
            image = Image.open(f"./dataset_sub/{image_path}").convert("RGB")
        else:
            image = Image.open(f"./dataset/{image_path}").convert("RGB")

        image = transform(image)

        return image, label

def create_model():
    """
    Creates a model using the EfficientNet-B7 architecture.

    This function initializes an EfficientNet-B7 model with pre-trained weights and modifies its classifier to match the number of output classes.

    Returns:
        model: A PyTorch model object.
    """
    from torch import nn
    from torchvision.models import efficientnet_b3
    model = efficientnet_b3(pretrained = True)

    model.classifier = nn.Sequential(
        nn.Linear(model.classifier[1].in_features, 128),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(128, len(labels.keys()))
    )

    return model

# Function to extract label from filename or directory structure
def extract_label(image_name):
    
    import re
    label = re.sub(r'\d+', '', image_name).replace(".jpg", "").replace(".png", "")
    return label

# Update dataset processing to include subclasses
def training_labels(train_size, test_size, dataset_folder="dataset"):
    """
    Creates training, validation, and testing sets from the dataset.

    This function randomly splits the dataset into training, validation, and testing sets based on the specified percentages. It also converts the labels into one-hot vectors for easier processing.

    Parameters:
    train_size (int): The percentage of the dataset to use for training.
    test_size (int): The percentage of the dataset to use for testing.

    Returns:
    tuple: A tuple containing three dictionaries: training_images, validation_images, and testing_images.
    """

    from os import listdir
    from random import randint
    from torch import zeros
    training_images, validation_images, testing_images = {}, {}, {}
    
    if dataset_folder == "dataset":
        for index, key in enumerate(list(labels.keys())):
            one_hot_vector = zeros(len(labels))
            one_hot_vector[index] = 1

            labels[key] = one_hot_vector
            
        for path in listdir(dataset_folder):
            label = extract_label(path)
            if label is None:
                continue
            set_choice = randint(0, 100)
            if set_choice < train_size:
                training_images[path] =  labels[label.replace("_", "")]
            elif set_choice >  test_size:
                validation_images[path] =  labels[label.replace("_", "")]
            else:
                testing_images[path] =  labels[label.replace("_", "")]
                
        print(f"Training set size: {len(training_images)}")
        print(f"Validation set size: {len(validation_images)}")
        print(f"Testing set size: {len(testing_images)}")
        
    else:
        for category in subclasses.keys():
            category_path = os.path.join(dataset_folder, category)
            if not os.path.exists(category_path):
                continue
            for subcategory in subclasses[category]:
                subcategory_path = os.path.join(category_path, subcategory)
                
                if not os.path.exists(subcategory_path):
                    continue
                for path in listdir(subcategory_path):
                    image_path = os.path.join(category, subcategory, path)
                    label, sublabel = labels[category], subclasses[category].index(subcategory)
                    set_choice = randint(0, 100)
                    if set_choice < train_size:
                        training_images[image_path] = sublabel
                    elif set_choice > test_size:
                        validation_images[image_path] = sublabel
                    else:
                        testing_images[image_path] = sublabel
                        
            print(f"Sublevel Training size: {len(training_images)}")
            print(f"Sublevel Validation size: {len(validation_images)}")
            print(f"Sublevel Testing size: {len(testing_images)}")
            
    return training_images, validation_images, testing_images

def run_training_pipeline(sub=False):
    """
    Runs the training pipeline for the model. This function prepares the dataset by splitting it into training, validation, and testing sets, initializes the model, and sets up the data loaders. It also defines the loss function and optimizer, and manages the training process over a specified number of epochs, tracking the training and validation losses to prevent overfitting.

    Parameters:
    None

    Returns:
    None
    """
    import torch
    from json import loads, dump
    from torch.utils.data import DataLoader
    from torch import nn
    from torch.optim import Adam
    import numpy as np

    global labels
    confusion_matrix = torch.zeros(len(labels.keys()), len(labels.keys()))
    # sub_confusion_matrix = torch.zeros(len(subclasses.keys()), len(subclasses.values()))
    with open('parameters.json') as f:
        parameters = loads(f.read())

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Creating model")
    model = create_model()
    print("Creating training, validation, and testing sets")
    training_set, validation_set, testing_set = training_labels(parameters['train_size'], parameters['test_size'])
    
    if sub:
        sub_training_set, sub_validation_set, sub_testing_set = training_labels(70, 90, "dataset_sub")
        
    model.to(device)
    
    print("Creating training and validation data loaders")
    dataset = ImageDataset(training_set)
    train_loader = DataLoader(dataset, batch_size=parameters['batch_size'], shuffle=True)

    dataset = ImageDataset(validation_set)
    val_loader = DataLoader(dataset, batch_size=parameters['batch_size'], shuffle=True)

    # If we want subclasses
    if sub:
        sub_training_dataset = ImageDataset(sub_training_set, True)
        sub_train_loader = DataLoader(sub_training_dataset, batch_size=parameters['batch_size'], shuffle=True)
        sub_dataset = ImageDataset(sub_validation_set, True)
        sub_val_loader = DataLoader(sub_dataset, batch_size=parameters['batch_size'], shuffle=True)

    loss_function = nn.CrossEntropyLoss() #Applies softmax on its own, dont need to apply it to data
    optim = Adam(model.parameters(), lr=parameters['learning_rate'])

    best_loss = float('inf')
    epochs_no_improve = 0

    print("Creating training info")
    if not parameters['load_model']:
        training_info = {
            'train_loss': [],
            'val_loss': [],
            'test_loss': []
        }
        sub_training_info = {
            'train_loss': [],
            'val_loss': [],
            'test_loss': []
        }
    else:
        training_info = loads('training_data.json')
        

    if bool(parameters['train_model']):
        print("Training model")
        for epoch in range(parameters['epochs']):
            torch.cuda.empty_cache()
            torch.autograd.set_detect_anomaly(True)
            training_loss = 0
            sub_training_loss = 0
            # Training loop
            model.train()
            for images, labels in train_loader:
                images = images.to(device)
                labels = labels.to(device)
                optim.zero_grad()

                outputs = model(images)

                loss = loss_function(outputs, labels)

                loss.backward()
                optim.step()

                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                training_loss += loss.item()
                del images
                del labels
                del outputs
                torch.cuda.empty_cache()
            
            # For subclass training 
            if sub:
                for images, labels in sub_train_loader:
                    images = images.to(device)
                    labels = labels.to(device)
                    optim.zero_grad()

                    outputs = model(images)

                    loss = loss_function(outputs, labels)

                    loss.backward()
                    optim.step()

                    nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                    training_loss += loss.item()
                    del images
                    del labels
                    del outputs
                    torch.cuda.empty_cache()

            print(f"Epoch: {epoch}   |  Training Loss: {training_loss / len(train_loader)}")
            if sub:
                print(f"Epoch: {epoch}   |  Sub Training Loss: {sub_training_loss / len(sub_train_loader)}")
                sub_training_info['train_loss'].append(sub_training_loss / len(sub_train_loader))
                
            training_info['train_loss'].append(training_loss / len(train_loader))
            

            # Validation loop
            validation_loss = 0
            sub_validation_loss = 0
            model.eval()
            with torch.no_grad():
                for images, labels in val_loader:
                    images = images.to(device)
                    labels = labels.to(device)
                    outputs = model(images)
                    loss = loss_function(outputs, labels)
                    validation_loss += loss.item()
                    del images
                    del labels
                    del outputs
                    torch.cuda.empty_cache()
                
                if sub:
                    for images, labels in sub_val_loader:
                        images = images.to(device)
                        labels = labels.to(device)
                        outputs = model(images)
                        loss = loss_function(outputs, labels)
                        sub_validation_loss += loss.item()
                        del images
                        del labels
                        del outputs
                        torch.cuda.empty_cache()

            training_info['val_loss'].append(validation_loss / len(val_loader))
            print(f"Epoch: {epoch}   |  Validation Loss: {validation_loss / len(val_loader)}")

            if sub:
                sub_training_info['val_loss'].append(sub_validation_loss / len(sub_val_loader))
                print(f"Epoch: {epoch}   |  Sub Validation Loss: {sub_validation_loss / len(sub_val_loader)}")

            if epoch % 2 == 0:
                torch.save(model.state_dict(), f'model_{epoch}.pth')

            if validation_loss < best_loss:
                best_loss = validation_loss
                epochs_no_improve = 0
                torch.save(model.state_dict(), f'best_model_epoch_{epoch}.pth')
            else:
                epochs_no_improve += 1

            if epochs_no_improve >= parameters['patience']:
                break

    print("Creating testing data loader")
    dataset = ImageDataset(testing_set)
    test_loader = DataLoader(dataset, batch_size=parameters['batch_size'], shuffle=True)
    
    if sub:
        print("Creating Sub testing data loader")
        sub_test_dataset = ImageDataset(sub_testing_set, True)
        sub_test_loader = DataLoader(sub_test_dataset, batch_size=parameters['batch_size'], shuffle=True)

    # Testing loop
    print("Evaluating model")
    model.eval()
    testing_loss = 0
    sub_testing_loss = 0
    with torch.no_grad():
        for images, labels in test_loader:

            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = loss_function(outputs, labels)
            testing_loss += loss.item()

            outputs = outputs.cpu().numpy()
            labels = labels.cpu().numpy()

            for i in range(len(outputs)):
                target_label = np.argmax(labels[i])  # Find the index of the column that contains the
                predicted_label = np.argmax(outputs[i])  # Find the index of the column that contains the 1
                confusion_matrix[target_label][predicted_label] += 1

            del images
            del labels
            del outputs
            torch.cuda.empty_cache()
        
        if sub:
            for images, labels in sub_test_loader:

                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                loss = loss_function(outputs, labels)
                sub_testing_loss += loss.item()

                outputs = outputs.cpu().numpy()
                labels = labels.cpu().numpy()

                for i in range(len(outputs)):
                    target_label = np.argmax(labels[i])  # Find the index of the column that contains the
                    predicted_label = np.argmax(outputs[i])  # Find the index of the column that contains the 1
                    # sub_confusion_matrix[target_label][predicted_label] += 1

                del images
                del labels
                del outputs
                torch.cuda.empty_cache()

    print(confusion_matrix)
    plot_heatmap(confusion_matrix)

    # if sub:
    #     print(sub_confusion_matrix)
        
    print(f"Testing Loss: {testing_loss / len(test_loader)}")
    training_info['test_loss'].append(testing_loss / len(test_loader))
    
    if sub:
        print(f"Sub Testing Loss: {sub_training_loss / len(sub_test_loader)}")
        sub_training_info['test_loss'].append(sub_training_loss / len(sub_test_loader))

    with open("training_data.json", "w") as json_file:
        dump(training_info, json_file, indent=4)


def plot_heatmap(matrix):
    """
    Plots a heatmap of the confusion matrix.

    This function creates a heatmap visualization of the confusion matrix, which shows the frequency of correct and incorrect predictions for each class.

    Parameters:
        matrix (numpy.ndarray): The confusion matrix to plot.

    Returns:
        None
    """
    import numpy as np
    global label_list

    plt.figure(figsize=(10, 8))
    plt.imshow(matrix, cmap='hot', interpolation='nearest')
    plt.colorbar()
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.xticks(ticks=np.arange(len(label_list)), labels=label_list)
    plt.yticks(ticks=np.arange(len(label_list)), labels=label_list)
    plt.savefig('confusion_matrix.png')
    plt.close()
    
def move_images():
    """
    Moves images from the original dataset to a new dataset with a 50% chance.

    This function randomly moves images from the original dataset to a new dataset
    with a 50% chance. It ensures that the new dataset is created if it doesn't exist.

    Returns:
        None
    """
    from os import listdir, mkdir
    from os.path import exists
    from random import randint
    from shutil import move

    original = './dataset'
    new = './dataset_new'

    if not exists(new):
        mkdir(new)

    for image in listdir(original):
        if randint(0, 100) < 50:
            move(f'{original}/{image}', f'{new}/{image}')

#move_images() #Uncomment to move images to new dataset - Should only be done once. This allows for the model to be trained on a smaller dataset, and use the remaining images for fine tuning training afterwards using the segmentation model's output as a mask

run_training_pipeline(True) # Change to False To remove subclasses or just remove True