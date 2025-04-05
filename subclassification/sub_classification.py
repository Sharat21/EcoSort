from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt

labels = {
    'PET': 0,
    'HDPE': 1,
    'PVC': 2,
    'LDPE': 3,
    'PP' : 4,
    'PS' : 5
}

label_list = list(labels.keys())

transform = transforms.Compose([
            transforms.Resize((300, 300)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
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
        image = Image.open(f"./dataset_subclass/{image_path}").convert("RGB")

        image = transform(image)

        return image, label

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

    directory = './dataset_subclass/'
    training_images, validation_images, testing_images = {}, {}, {}

    for index, key in enumerate(list(labels.keys())):
        one_hot_vector = zeros(len(labels))
        one_hot_vector[index] = 1

        labels[key] = one_hot_vector

    for path in listdir(directory):
        label = re.sub(r'\d+', '', path).replace(".jpg", "").replace(".png", "")
        set = randint(0, 100)
        if set < train_size:
            training_images[path] = labels[label.replace("_", "")]

        elif set > test_size:
            testing_images[path] = labels[label.replace("_", "")]

        else:
            validation_images[path] = labels[label.replace("_", "")]
    
    print(f"Training set size: {len(training_images)}")
    print(f"Validation set size: {len(validation_images)}")
    print(f"Testing set size: {len(testing_images)}")

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
    confusion_matrix = torch.zeros(len(labels.keys()), len(labels.keys()))

    with open('parameters.json') as f:
        parameters = loads(f.read())

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Creating model")
    model = create_model()
    print("Creating training, validation, and testing sets")
    training_set, validation_set, testing_set = training_labels(parameters['train_size'], parameters['test_size'])
    model.to(device)

    print("Creating training and validation data loaders")
    dataset = ImageDataset(training_set)
    train_loader = DataLoader(dataset, batch_size=parameters['batch_size'], shuffle=True)
    dataset = ImageDataset(validation_set)
    val_loader = DataLoader(dataset, batch_size=parameters['batch_size'], shuffle=True)


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
                torch.save(model.state_dict(), f'best_model_epoch_{epoch}.pth')
            else:
                epochs_no_improve += 1

            if epochs_no_improve >= parameters['patience']:
                break

    print("Creating testing data loader")
    dataset = ImageDataset(testing_set)
    test_loader = DataLoader(dataset, batch_size=parameters['batch_size'], shuffle=True)

    # Testing loop
    print("Evaluating model")
    model.eval()
    testing_loss = 0
    correct_predictions = 0  # Track correct predictions
    total_predictions = 0  # Track total samples
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
                if target_label == predicted_label:
                    correct_predictions += 1  # Increment correct predictions count

                confusion_matrix[target_label][predicted_label] += 1
            
            total_predictions += len(outputs)
            del images
            del labels
            del outputs
            torch.cuda.empty_cache()

    print(confusion_matrix)
    plot_heatmap(confusion_matrix)
    
    test_accuracy = (correct_predictions / total_predictions) * 100
    print(f"Test Accuracy: {test_accuracy:.2f}%")
    print(f"Testing Loss: {testing_loss / len(test_loader)}")
    training_info['test_loss'].append(testing_loss / len(test_loader))

    with open("training_data.json", "w") as json_file:
        dump(training_info, json_file, indent=4)


run_training_pipeline()