import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

class DigitCNN(nn.Module):
    def __init__(self):
        super(DigitCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64 * 3 * 3, 256)  # Adjust input size after convolution and pooling
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 10)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.batch_norm1 = nn.BatchNorm2d(32)
        self.batch_norm2 = nn.BatchNorm2d(64)
        self.batch_norm_fc1 = nn.BatchNorm1d(256)

    def forward(self, x):
        x = torch.relu(self.batch_norm1(self.conv1(x)))
        x = torch.max_pool2d(x, 2, 2)
        x = torch.relu(self.batch_norm2(self.conv2(x)))
        x = torch.max_pool2d(x, 2, 2)
        x = torch.relu(self.conv3(x))
        x = torch.max_pool2d(x, 2, 2)
        x = x.view(x.size(0), -1)  # Flatten the tensor for the fully connected layer
        x = torch.relu(self.batch_norm_fc1(self.fc1(x)))
        x = self.dropout1(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)
        return torch.log_softmax(x, dim=1)

def limit_gpu_memory_usage(memory_fraction=0.5):
    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(memory_fraction)
        torch.cuda.empty_cache()

def train_and_save_model(model, device, epochs=10):
    transform = transforms.Compose([
        transforms.RandomRotation(10),  # Randomly rotate images for augmentation
        transforms.RandomAffine(0, shear=10, scale=(0.8, 1.2)),  # Apply shear and scaling
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adadelta(model.parameters())

    model.train()
    for epoch in range(epochs):
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            if batch_idx % 100 == 0:
                print(f'Train Epoch: {epoch} [{batch_idx*len(data)}/{len(train_loader.dataset)} ({100.*batch_idx/len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')

    torch.save(model.state_dict(), "digit_recognition_model.pth")
    print("Model saved as better_digit_recognition_model.pth")

if __name__ == "__main__":
    limit_gpu_memory_usage(memory_fraction=0.5)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DigitCNN().to(device)
    train_and_save_model(model, device)
