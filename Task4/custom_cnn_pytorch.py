import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import matplotlib.pyplot as plt
import time

start_time = time.time()  # Start time

MODEL_NAME = 'model_pytorch_1'
WEIGHTS_DIR = f'/ghome/group01/group01/project23-24-01/Task4/weights/{MODEL_NAME}.h5'
RESULTS_DIR = '/ghome/group01/group01/project23-24-01/Task4/results'
DATASET_DIR = '/ghome/mcv/datasets/C3/MIT_small_train_1'
DATASET_DIR_GLOBAL = '/ghome/mcv/datasets/C3/MIT_split'

NUM_CLASSES = 8
BATCH_SIZE = 64
IMG_SIZE = (256, 256)
EPOCHS = 80
# setting device on GPU if available, else CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)
print()
# Additional Info when using cuda
if device.type == 'cuda':
    print(torch.cuda.get_device_name(0))
    print('Memory Usage:')
    print('Allocated:', round(torch.cuda.memory_allocated(0) / 1024 ** 3, 1), 'GB')
    print('Cached:   ', round(torch.cuda.memory_reserved(0) / 1024 ** 3, 1), 'GB')


class CNNModel(nn.Module):
    def __init__(self, num_classes):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.batchnorm1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.batchnorm2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.batchnorm3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.batchnorm4 = nn.BatchNorm2d(256)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(256 * (IMG_SIZE[0] // 16) * (IMG_SIZE[1] // 16), 512)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.pool1(F.relu(self.batchnorm1(self.conv1(x))))
        x = self.pool2(F.relu(self.batchnorm2(self.conv2(x))))
        x = self.pool3(F.relu(self.batchnorm3(self.conv3(x))))
        x = self.pool4(F.relu(self.batchnorm4(self.conv4(x))))
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class DeviceDataLoader:
    def __init__(self, dataloader, device):
        self.dataloader = dataloader
        self.device = device

    def __iter__(self):
        for data in self.dataloader:
            inputs, labels = data
            yield inputs.to(self.device), labels.to(self.device)

    def __len__(self):
        return len(self.dataloader)


# Data loading
transform = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.ToTensor(),
])
train_dataset = ImageFolder(root=DATASET_DIR + '/train/', transform=transform)
val_dataset = ImageFolder(root=DATASET_DIR + '/test/', transform=transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

train_loader = DeviceDataLoader(train_loader, device)
val_loader = DeviceDataLoader(val_loader, device)

# Model, loss, and optimizer
model = CNNModel(num_classes=NUM_CLASSES)
model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Counting number of parameters
def get_n_params(model):
    pp=0
    for p in list(model.parameters()):
        nn=1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    return pp
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total Trainable Parameters: {total_params}")
print(f"Total Trainable Parameters: {get_n_params(model)}")
# Lists to store results for plotting
train_losses = []
val_losses = []
train_accuracies = []
val_accuracies = []

# Training loop
# for epoch in range(EPOCHS):
#     model.train()
#     correct_train = 0
#     total_train = 0
#     for inputs, labels in train_loader:
#         optimizer.zero_grad()
#         outputs = model(inputs)
#         loss = criterion(outputs, labels)
#         loss.backward()
#         optimizer.step()

#         _, predicted = torch.max(outputs.data, 1)
#         total_train += labels.size(0)
#         correct_train += (predicted == labels).sum().item()

#     train_accuracy = correct_train / total_train
#     train_accuracies.append(train_accuracy)

#     # Validation
#     model.eval()
#     with torch.no_grad():
#         val_loss = 0.0
#         correct_val = 0
#         total_val = 0
#         for inputs, labels in val_loader:
#             outputs = model(inputs)
#             loss = criterion(outputs, labels)
#             val_loss += loss.item()
#             _, predicted = torch.max(outputs.data, 1)
#             total_val += labels.size(0)
#             correct_val += (predicted == labels).sum().item()

#     val_accuracy = correct_val / total_val
#     val_accuracies.append(val_accuracy)

#     # Save results for plotting
#     train_losses.append(loss.item())
#     val_losses.append(val_loss / len(val_loader))

#     print(
#         f'Epoch {epoch + 1}/{EPOCHS}, Loss: {loss.item():.4f}, Train Accuracy: {train_accuracy:.4f}, Validation Accuracy: {val_accuracy:.4f}')
# end_time = time.time()  # End time
# training_time = end_time - start_time
# print(f'Training completed in {training_time:.2f} seconds')
# # Save the PyTorch model
# torch.save(model.state_dict(), WEIGHTS_DIR)

# # Plotting results
# plt.figure(figsize=(12, 6))

# # Plotting losses
# plt.subplot(1, 2, 1)
# plt.plot(train_losses, label='Train Loss')
# plt.plot(val_losses, label='Validation Loss')
# plt.title('Losses')
# plt.xlabel('Epochs')
# plt.ylabel('Loss')
# plt.legend()
# plt.grid()

# # Plotting accuracies
# plt.subplot(1, 2, 2)
# plt.plot(train_accuracies, label='Train Accuracy')
# plt.plot(val_accuracies, label='Validation Accuracy')
# plt.title('Accuracies')
# plt.xlabel('Epochs')
# plt.ylabel('Accuracy')
# plt.legend()
# plt.grid()

# # Save plots
# plt.tight_layout()
# plt.savefig(f'{RESULTS_DIR}/{MODEL_NAME}_loss_accuracy.jpg')
# plt.show()
