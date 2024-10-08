import torch
import torch.nn as nn
import torch.optim as optim
import os
import json
from tqdm import tqdm
import random
import argparse


class SingleLayerClassifier(nn.Module):
    """
    A linear classifier with a classification (Input: Layer last token embedding, Output: KV-Pair/Document ID).
    """

    def __init__(self, input_size, num_classes):
        super(SingleLayerClassifier, self).__init__()
        # Linear layer to transform inputs to outputs
        self.linear = nn.Linear(input_size, num_classes)

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
    # Compute the standard deviation of each feature, prevent division by zero
    std = X.std(dim=0, keepdim=True) + 1e-8
    normalized_X = (X - mean) / std  # Normalize features
    return normalized_X


def test(model, data):
    """
    Evaluate the model performance on the provided dataset.

    Args:
        model: SoftMax Linear Classifier to be tested.
        data: Data containing the test dataset.
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
                class_abs_diff[label.item()] += (prediction -
                                                 label).abs().item()
                if label == prediction:
                    class_correct[label.item()] += 1

            total_loss += loss.item()
            total_abs_diff += (predicted - Y).abs().sum().item()

    accuracy = 100 * correct / total if total > 0 else 0
    class_accuracies = []
    class_avg_abs_diff = {c: class_abs_diff[c] / class_counts[c]
                          if class_counts[c] > 0 else 0 for c in sorted(class_total.keys())}
    for c in sorted(class_total.keys()):
        class_acc = 100 * class_correct[c] / \
            class_total[c] if class_total[c] > 0 else 0
        class_accuracies.append(class_acc)

    return total_loss / len(data), accuracy, class_accuracies, total_abs_diff / total, class_avg_abs_diff


def train(model, data, epochs, lr):
    """
    Trains a SoftMax Linear Classifier using the provided data, number of epochs, and learning rate.

    Args:
        model: Classifier to be trained.
        data: Data containing the training dataset.
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
        class_correct = {label: 0 for label in num_classes}
        class_total = {label: 0 for label in num_classes}
        class_abs_diff = {label: 0 for label in num_classes}
        class_counts = {label: 0 for label in num_classes}

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
                class_total[label.item()] += 1
                class_counts[label.item()] += 1
                class_abs_diff[label.item()] += (prediction - label).abs().item()
                if label == prediction:
                    class_correct[label.item()] += 1

            total_loss += loss.item()
            total_abs_diff += (predicted - Y).abs().sum().item()

        accuracy = 100 * correct / total if total > 0 else 0

        class_avg_abs_diff = {c: class_abs_diff[c] / class_counts[c]
                              if class_counts[c] > 0 else 0 for c in sorted(class_total.keys())}
        class_accuracies = [100 * class_correct[c] / class_total[c]
                            if class_total[c] > 0 else 0 for c in sorted(class_total.keys())]

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
        loss, accuracy, class_accuracies, dist, class_dist = train(
            model, data, epochs=150, lr=0.005)
    else:
        model.eval()
        with torch.no_grad():
            loss, accuracy, class_accuracies, dist, class_dist = test(model, [
                                                                      (X, Y)])

    print(
        f"The overall acc is {accuracy}, the class acc is {class_accuracies}")
    return loss, accuracy, class_accuracies, dist, class_dist


def validate_tensors(tensor_list):
    for idx, item in enumerate(tensor_list):

        assert isinstance(
            item, torch.Tensor), f"Item at index {idx} is not a tensor: {type(item)}"
        if not isinstance(item, torch.Tensor):
            print(
                f"Non-tensor found at index {idx}: {item}, type: {type(item)}")


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


def process_data(data, indices, shuffle=False):
    X = []
    Y = []
    for index in indices:
        X += data[index]
        Y += ([indices.index(index)] * len(data[index]))

    if shuffle:
        shuffle_indices = list(range(len(X)))
        random.shuffle(shuffle_indices)
        X = [X[i] for i in shuffle_indices]
        Y = [Y[i] for i in shuffle_indices]
    validate_tensors(X)

    X = torch.stack(X, dim=0)
    Y = torch.tensor(Y)
    return X, Y


def main(data_path, output_folder):
    print(f'Processing {data_path}', flush=True)
    data = read_data(data_path, max_files=1)

    indices = sorted(list(data.keys()))
    print('Indices are:', indices)

    X, Y = process_data(data, indices)
    length = len(X)
    print(f"There are {length} data all together", flush=True)

    split_point = int(length * 0.8)
    X_train = X[:split_point]
    X_test = X[split_point:]
    Y_train = Y[:split_point]
    Y_test = Y[split_point:]

    emb_dim = X.shape[2]
    num_classes = len(indices)
    results_single = []

    for i in tqdm(range(X_train.shape[1])):
        print(f"Training on layer {i+1}", flush=True)
        Xi_train = X_train[:, i, :]
        Xi_test = X_test[:, i, :]
        model = SingleLayerClassifier(emb_dim, num_classes)

        train_results = classify(
            model, indices, Xi_train, Y_train, train_mode=True)
        test_results = classify(model, indices, Xi_test,
                                Y_test, train_mode=False)

        results_single.append({
            'layer_num': i,
            'train_loss': train_results[0],
            'train_accuracy': train_results[1],
            'train_class_accuracies': train_results[2],
            'train_distance': train_results[3],
            'train_class_distance': train_results[4],
            'test_loss': test_results[0],
            'test_accuracy': test_results[1],
            'test_class_accuracies': test_results[2],
            'test_distance': test_results[3],
            'test_class_distance': test_results[4]
        })

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    results_file_path = os.path.join(output_folder, f'probing-results.json')
    with open(results_file_path, 'w') as file:
        json.dump(results_single, file, indent=4)
    print(f'Results saved to {results_file_path}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process and classify data.')
    parser.add_argument('data_path', type=str, help='Path to the data file')
    parser.add_argument('output_folder', type=str,
                        help='Output folder for results')
    args = parser.parse_args()

    main(args.data_path, args.output_folder)
