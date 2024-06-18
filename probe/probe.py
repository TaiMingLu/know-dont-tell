import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
import json
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import sys
import random
from torch.utils.data import DataLoader, TensorDataset

class SingleLayerClassifier(nn.Module):
    """
    A linear classifier with a classification (Input: Layer last token embedding, Output: KV-Pair/Document ID).
    """
    def __init__(self, input_size, num_classes):
        super(SingleLayerClassifier, self).__init__()
        self.linear = nn.Linear(input_size, num_classes)  # Linear layer to transform inputs to outputs

    def forward(self, x):
        return self.linear(x)

def normalize_data(X):
    """
    Normalize the input data to have zero mean and unit variance.

    Args:
        X (Tensor): The input data tensor to be normalized.

    Returns:
        Tensor: The normalized data tensor.
    """
    mean = X.mean(dim=0, keepdim=True)  # Compute the mean of each feature
    std = X.std(dim=0, keepdim=True) + 1e-8  # Compute the standard deviation of each feature, prevent division by zero
    normalized_X = (X - mean) / std  # Normalize features
    return normalized_X

def test(model, data):
    """
    Evaluate the model performance on the provided dataset.

    Args:
        model: SoftMax Linear Classifier to be tested.
        data: A DataLoader containing the test dataset.
    Returns:
        A tuple containing the average loss, overall accuracy, per-class accuracies,
        average absolute difference across all predictions, and average absolute difference per class.
    """
    criterion = torch.nn.CrossEntropyLoss()
    model.eval()
    total_abs_diff = 0
    total_loss = 0
    correct = 0
    total = 0
    class_correct = dict()
    class_total = dict()
    class_abs_diff = dict()
    class_counts = dict()
    
    with torch.no_grad():
        for X, Y in data:
            outputs = model(X)
            loss = criterion(outputs, Y)

            _, predicted = torch.max(outputs.data, 1)
            total += Y.size(0)
            correct += (predicted == Y).sum().item()

            for label, prediction in zip(Y, predicted):
                if label.item() not in class_total:
                    class_total[label.item()] = 0
                    class_correct[label.item()] = 0
                    class_abs_diff[label.item()] = 0
                    class_counts[label.item()] = 0
                class_total[label.item()] += 1
                class_counts[label.item()] += 1
                class_abs_diff[label.item()] += (prediction - label).abs().item()
                if label == prediction:
                    class_correct[label.item()] += 1

            total_loss += loss.item()
            total_abs_diff += (predicted - Y).abs().sum().item()

    accuracy = 100 * correct / total if total > 0 else 0
    class_accuracies = []
    class_avg_abs_diff = {c: class_abs_diff[c] / class_counts[c] if class_counts[c] > 0 else 0 for c in sorted(class_total.keys())}
    for c in sorted(class_total.keys()):
        class_acc = 100 * class_correct[c] / class_total[c] if class_total[c] > 0 else 0
        class_accuracies.append(class_acc)

        
    return total_loss / len(data), accuracy, class_accuracies, total_abs_diff / total, class_avg_abs_diff

def train(model, data, epochs, lr):
    """
    Trains a SoftMax Linear Classifier using the provided data, number of epochs, and learning rate.

    Args:
        model: Classifier to be trained.
        data: A DataLoader containing the training dataset.
        epochs (int): The number of times to iterate over the entire dataset.
        lr (float): Learning rate for the optimizer.

    Returns:
        A tuple containing the average loss, overall accuracy, per-class accuracies,
        average absolute difference across all predictions, and average absolute difference per class.
    """
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-3)

    model.train()
    class_accuracies = []


    for epoch in range(epochs):
        total_abs_diff = 0
        total_loss = 0
        correct = 0
        total = 0
        class_correct = dict()
        class_total = dict()
        class_abs_diff = dict()
        class_counts = dict()

        for X, Y in data:
            optimizer.zero_grad()
            outputs = model(X)
            loss = criterion(outputs, Y)
            loss.backward()
            optimizer.step()

            _, predicted = torch.max(outputs.data, 1)
            total += Y.size(0)
            correct += (predicted == Y).sum().item()

            for label, prediction in zip(Y, predicted):
                if label.item() not in class_total:
                    class_total[label.item()] = 0
                    class_correct[label.item()] = 0
                    class_abs_diff[label.item()] = 0
                    class_counts[label.item()] = 0
                class_total[label.item()] += 1
                class_counts[label.item()] += 1
                class_abs_diff[label.item()] += (prediction - label).abs().item()
                if label == prediction:
                    class_correct[label.item()] += 1


            total_loss += loss.item()
            total_abs_diff += (predicted - Y).abs().sum().item()

        accuracy = 100 * correct / total if total > 0 else 0

        class_avg_abs_diff = {c: class_abs_diff[c] / class_counts[c] if class_counts[c] > 0 else 0 for c in sorted(class_total.keys())}
        class_accuracies = [100 * class_correct[c] / class_total[c] if class_total[c] > 0 else 0 for c in sorted(class_total.keys())]

        
    return total_loss / len(data), accuracy, class_accuracies, total_abs_diff / total, class_avg_abs_diff


def classify(model, num_classes, X, Y, train_mode=True):
    """
    Classify or train the model based on the given dataset and mode.

    Args:
        model: The neural network model to be trained or tested.
        num_classes: The number of different IDs.
        X: Input features tensor.
        Y: Labels tensor.
        train_mode (bool): Flag to determine whether to train the model or classify using the existing model.

    Returns:
        A tuple containing the loss, overall accuracy, per-class accuracies, total distance, and per-class distance.
    """
    X = normalize_data(X)
    emb_dim = (X.shape[1])

    print(f"X shape: {X.shape}, Y shape: {Y.shape}, emd_dim: {emb_dim}, num_classes: {num_classes}, now train is set to be {train_mode}")

    if train_mode:
        data = [(X, Y)]
        loss, accuracy, class_accuracies, dist, class_dist = train(model, data, epochs=150, lr=0.005)
    else:
        model.eval()
        with torch.no_grad():
            loss, accuracy, class_accuracies, dist, class_dist = test(model, [(X, Y)])

    print(f"The overall acc is {accuracy}, the class acc is {class_accuracies}")
    return loss, accuracy, class_accuracies, dist, class_dist

def validate_tensors(tensor_list):
    for idx, item in enumerate(tensor_list):

        assert isinstance(item, torch.Tensor), f"Item at index {idx} is not a tensor: {type(item)}"
        if not isinstance(item, torch.Tensor):
            print(f"Non-tensor found at index {idx}: {item}, type: {type(item)}")

def read_data(path, max_files=None):
    data_dict = {}
    file_count = 0
    
    for filename in os.listdir(path):
        if max_files is not None and file_count >= max_files:
            break
        
        file_path = os.path.join(path, filename)
        data = torch.load(file_path)

        print(f'Data read from {filename}, processing', flush=True)
        transposed_data = list(zip(*data))
        training_data = [list(transposed_data[0]), list(transposed_data[2])]
        del data
        del transposed_data

        for i in range(len(training_data[0])):
            mat = training_data[0][i]
            index = training_data[1][i]
            if index not in data_dict:
                data_dict[index] = []
            data_dict[index].append(mat)

        del training_data

        file_count += 1

    return data_dict

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python script.py <data_path>")
        sys.exit(1)

    data_path = sys.argv[1]

    print(f'Processing {data_path}', flush=True)

    data = read_data(data_path, max_files=1)

    indices = sorted(list(data.keys())) 
    print('Indices are:', indices)

    results_single = []
    X = []
    Y = []

    for index in indices:
        X += data[index]
        print(f"The length of {index} is {len(data[index])}")
        Y += ([indices.index(index)] * len(data[index]))

        print(f"The length of whole data is: {len(X)}", flush=True)
        shuffle_indices = list(range(len(X)))
        random.shuffle(shuffle_indices)
        X = [X[i] for i in shuffle_indices]
        Y = [Y[i] for i in shuffle_indices]
        validate_tensors(X)


        X = torch.stack(X, dim=0)
        Y = torch.tensor(Y)

        length = len(X)
        print(f"THere are {length} data all togethter", flush=True)
        split_point = int(length * 0.8)
        X_train = X[:split_point]
        X_test = X[split_point:]
        Y_train = Y[:split_point]
        Y_test = Y[split_point:]

        emb_dim = (X.shape[2])
        num_classes = len(indices)

        for i in range(X_train.shape[1]):
            print(f"Training on layer {i+1}", flush=True)
            Xi_train = X_train[:, i, :]
            Xi_test = X_test[:, i, :]
            model = SingleLayerClassifier(emb_dim, num_classes)

            loss_train, accuracy_train, class_accuracies_train, dist_train, class_dist_train = classify(model, indices, Xi_train, Y_train, train_mode=True)
            loss_test, accuracy_test, class_accuracies_test, dist_test, class_dist_test = classify(model, indices, Xi_test, Y_test, train_mode=False)
            results_single.append({
                'layer_num': i,
                'train_loss': loss_train,
                'train_accuracy': accuracy_train,
                'train_class_accuracies': class_accuracies_train,
                'train_distance':dist_train,
                'train_class_distance': class_dist_train, 
                'test_loss': loss_test,
                'test_accuracy': accuracy_test,
                'test_class_accuracies': class_accuracies_test,
                'test_distance': dist_test,
                'test_class_distance': class_dist_test,  
            })


        with open(f'single_probing_results_{len(indices)}_{indices[-1]+1}-mistral-it-5.json', 'w') as file:
            json.dump(results_single, file, indent=4)