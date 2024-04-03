import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import precision_score, recall_score, accuracy_score, confusion_matrix
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn

class DNN(nn.Module):
    def __init__(self):
        super(DNN, self).__init__()
        self.fc1 = nn.Linear(16*16, 320)
        self.fc2 = nn.Linear(320, 50)
        self.fc3 = nn.Linear(50, 10)
        self.dropout = nn.Dropout(0.5)  # Adding dropout layer

    def forward(self, x):
        x = x.view(-1, 16*16)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)  # Applying dropout
        x = F.relu(self.fc2(x))
        x = self.dropout(x)  # Applying dropout
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, 5)
        self.conv2 = nn.Conv2d(10, 20, 5)
        self.maxpool = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(20, 50)
        self.fc2 = nn.Linear(50, 10)
        self.dropout = nn.Dropout(0.5)  # Adding dropout layer

    def forward(self, x):
        x = F.relu(self.maxpool(self.conv1(x)))
        x = self.dropout(x)  # Applying dropout
        x = F.relu(self.maxpool(self.conv2(x)))
        x = x.view(-1, 20)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)  # Applying dropout
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)



class Dataset:
    def read(self):
        transform = transforms.Compose([
            transforms.Resize((16,16)),  # Resize USPS images to 16x16
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

        train_dataset = datasets.USPS(root='./usps', train=True, download=True, transform=transform)
        test_dataset = datasets.USPS(root='./usps', train=False, download=True, transform=transform)

        train_loader = DataLoader(train_dataset, batch_size=1000, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=True)
        return train_loader, test_loader

def train_model(model, train_loader, optimizer, epoch, writer):
    model.train()
    for batch_idx, (data, label) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(data)
        train_loss = F.nll_loss(output, label)
        train_loss.backward()
        optimizer.step()
        if batch_idx % 10 == 0:
            print('Train Epoch: {}, Loss: {:.6f}'.format(epoch, train_loss.item()))
            writer.add_scalar('Loss/train', train_loss.item(), epoch * len(train_loader) + batch_idx)

def test_model(model, test_loader, writer, epoch):
    model.eval()
    test_loss = 0
    correct = 0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for data, label in test_loader:
            output = model(data)
            test_loss += F.nll_loss(output, label, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(label.view_as(pred)).sum().item()
            all_preds.extend(pred.numpy())
            all_labels.extend(label.numpy())

    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    writer.add_scalar('Loss/test', test_loss, epoch)
    writer.add_scalar('Accuracy/test', accuracy, epoch)
    return test_loss, accuracy, all_preds, all_labels

def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], fmt),
                     ha="center", va="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


# Function to calculate precision, recall, and confusion matrix
def calculate_metrics(predictions, labels):
    precision = precision_score(labels, predictions, average='macro', zero_division=0)
    recall = recall_score(labels, predictions, average='macro', zero_division=0)
    accuracy = accuracy_score(labels, predictions)
    cm = confusion_matrix(labels, predictions)
    return precision, recall, accuracy, cm


# Example values for demonstration
epochs = 10  # Number of epochs

# Instantiate models
cnn_model = CNN()
dnn_model = DNN()

# Define optimizers
cnn_optimizer = optim.SGD(cnn_model.parameters(), lr=0.01, momentum=0.9)
dnn_optimizer = optim.SGD(dnn_model.parameters(), lr=0.01, momentum=0.9)

# Load data
train_loader, test_loader = Dataset().read()

# Set up TensorBoard writer
cnn_writer = SummaryWriter('logs/cnn')
dnn_writer = SummaryWriter('logs/dnn')

# Training and evaluation loop for CNN model
for epoch in range(epochs):
    train_model(cnn_model, train_loader, cnn_optimizer, epoch, cnn_writer)
    test_loss, accuracy, cnn_preds, cnn_labels = test_model(cnn_model, test_loader, cnn_writer, epoch)
    print('\nCNN Test set: Average loss: {:.4f}, Accuracy: {:.2f}%\n'.format(test_loss, accuracy))

# Training and evaluation loop for DNN model
for epoch in range(epochs):
    train_model(dnn_model, train_loader, dnn_optimizer, epoch, dnn_writer)
    test_loss, accuracy, dnn_preds, dnn_labels = test_model(dnn_model, test_loader, dnn_writer, epoch)
    print('\nDNN Test set: Average loss: {:.4f}, Accuracy: {:.2f}%\n'.format(test_loss, accuracy))

# Compare accuracies
cnn_accuracy = sum(np.array(cnn_preds) == np.array(cnn_labels)) / len(cnn_preds)
dnn_accuracy = sum(np.array(dnn_preds) == np.array(dnn_labels)) / len(dnn_preds)

# Calculate precision, recall, and confusion matrix
cnn_precision, cnn_recall, _, cnn_cm = calculate_metrics(cnn_preds, cnn_labels)
dnn_precision, dnn_recall, _, dnn_cm = calculate_metrics(dnn_preds, dnn_labels)

# Print metrics for both models
print("CNN Accuracy:", cnn_accuracy)
print("CNN Precision:", cnn_precision)
print("CNN Recall:", cnn_recall)
print("CNN Confusion Matrix:")
print(cnn_cm)

print("DNN Accuracy:", dnn_accuracy)
print("DNN Precision:", dnn_precision)
print("DNN Recall:", dnn_recall)
print("DNN Confusion Matrix:")
print(dnn_cm)

# Close TensorBoard writers
cnn_writer.close()
dnn_writer.close()

