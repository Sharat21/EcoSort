from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt

labels = {
    'cardboard' : 0,
    'metal' : 1,
    'plastic' : 2,
    'trash' : 3,
    'glass' : 4,
    'paper' : 5,
    'biodegradeble' : 6
}

label_list = list(labels.keys())


transform = transforms.Compose([
            transforms.Resize((300, 300)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) #Same normalization method used originally on efficient net -> Better for fine tuning
])

class ImageDataset(Dataset):
    def __init__(self, images):

        self.images = images
    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_path = list(self.images.keys())[idx]
        label = self.images[image_path]

        # Open the image
        image = Image.open(f"./dataset_segmented/{image_path}").convert("RGB")

        image = transform(image)

        return image, label

def plot_heatmap(matrix, name):
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
    plt.savefig(name)
    plt.close()
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


def training_labels(train_size, test_size):
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
    import re
    from random import randint
    from torch import zeros

    directory = './dataset_segmented'
    training_images, validation_images, testing_images = {}, {}, {}

    for index, key in enumerate(list(labels.keys())):
        one_hot_vector = zeros(len(labels))
        one_hot_vector[index] = 1

        labels[key] = one_hot_vector


    for path in listdir(directory):

        label = path.split('_')[0]
        set = randint(0, 100)
        if set < train_size:
            training_images[path] = labels[label]

        elif set > test_size:
            testing_images[path] = labels[label]

        else:
            validation_images[path] = labels[label]

    return training_images, validation_images, testing_images



def run_training_pipeline():
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
    confusion_matrix_original = torch.zeros(len(labels.keys()), len(labels.keys()))
    confusion_matrix_fine_tuned = torch.zeros(len(labels.keys()), len(labels.keys()))

    with open('parameters.json') as f:
        parameters = loads(f.read())

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Creating model")
    model = create_model()
    model.load_state_dict(torch.load('classification_model.pth'))
    print("Creating training, validation, and testing sets")
    training_set, validation_set, testing_set = training_labels(parameters['train_size'], parameters['test_size'])
    model.to(device)

    loss_function = nn.CrossEntropyLoss() #Applies softmax on its own, dont need to apply it to data
    optim = Adam(model.parameters(), lr=parameters['learning_rate'])

    print("Creating training and validation data loaders")
    dataset = ImageDataset(training_set)
    train_loader = DataLoader(dataset, batch_size=parameters['batch_size'], shuffle=True)
    dataset = ImageDataset(validation_set)
    val_loader = DataLoader(dataset, batch_size=parameters['batch_size'], shuffle=True)


    best_loss = float('inf')
    epochs_no_improve = 0

    
    print("Creating training info")
    if not parameters['load_model']:
        training_info = {
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

            print(f"Epoch: {epoch}   |  Training Loss: {training_loss / len(train_loader)}")

            training_info['train_loss'].append(training_loss / len(train_loader))

            # Validation loop
            validation_loss = 0
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

            training_info['val_loss'].append(validation_loss / len(val_loader))
            print(f"Epoch: {epoch}   |  Validation Loss: {validation_loss / len(val_loader)}")

            if epoch % 2 == 0:
                torch.save(model.state_dict(), f'model_{epoch}.pth')

            if validation_loss < best_loss:
                best_loss = validation_loss
                epochs_no_improve = 0
                torch.save(model.state_dict(), f'classification_model_epoch_{epoch}.pth')
                classification_model = model.state_dict()
            else:
                epochs_no_improve += 1

            if epochs_no_improve >= parameters['patience']:
                break

    torch.save(classification_model, 'classification_model_fine_tuned.pth')


    model.load_state_dict(torch.load('classification_model_fine_tuned.pth'))

    print("Creating testing data loader")
    dataset = ImageDataset(testing_set)
    test_loader = DataLoader(dataset, batch_size=parameters['batch_size'], shuffle=True)

    # Testing loop using fine tuned weights
    print("Testing model with fine tuned weights")
    print("Evaluating model")
    testing_loss = 0
    correct = 0
    total = 0
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
                confusion_matrix_fine_tuned[target_label][predicted_label] += 1
                if target_label == predicted_label: correct += 1
                total += 1
            del images
            del labels
            del outputs
            torch.cuda.empty_cache()

    print(f"Accuracy: {(correct / total) * 100}")
    print(confusion_matrix_fine_tuned)
    plot_heatmap(confusion_matrix_fine_tuned, 'confusion_matrix_fine_tuned.png')

    print(f"Testing Loss: {testing_loss / len(test_loader)}")

    model.load_state_dict(torch.load('classification_model.pth'))
    # Testing loop using original weights
    print("Testing model with original weights")
    print("Evaluating model")
    testing_loss = 0
    correct = 0
    total = 0
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
                confusion_matrix_original[target_label][predicted_label] += 1
                if target_label == predicted_label: correct += 1
                total += 1
            del images
            del labels
            del outputs
            torch.cuda.empty_cache()

    print(f"Accuracy: {(correct / total) * 100}")
    print(confusion_matrix_original)
    plot_heatmap(confusion_matrix_original, 'confusion_matrix_original.png')

    print(f"Testing Loss: {testing_loss / len(test_loader)}")


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

    original = './dataset_segmented'
    new = './dataset_new'

    if not exists(new):
        mkdir(new)

    for image in listdir(original):
        if randint(0, 100) < 50:
            move(f'{original}/{image}', f'{new}/{image}')


def inference(image):
    """
    Performs inference on a new image using the trained model.

    This function loads the trained model, preprocesses the new image, and predicts its class.

    Returns:
        None
    """
    import torch
    import numpy as np
    import torchvision.transforms as transforms
    model = create_model()
    model.load_state_dict(torch.load('classification_model.pth'))
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    image = transform(image).unsqueeze(0).to(device)  # Apply transform, add batch dimension, and move to GPU

    output = model(image)
    output = output.cpu().detach().numpy()
    prediction = np.argmax(output)
    return label_list[prediction]

#move_images() #Uncomment to move images to new dataset - Should only be done once. This allows for the model to be trained on a smaller dataset, and use the remaining images for fine tuning training afterwards using the segmentation model's output as a mask
#run_training_pipeline()
