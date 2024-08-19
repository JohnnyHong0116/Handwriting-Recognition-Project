import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Path to the extracted images
extracted_images_dir = './ML_Dataset/Math Dataset/extracted_images'

class MathSymbolCNN(nn.Module):
    def __init__(self, num_classes):
        super(MathSymbolCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.fc1 = nn.Linear(64 * 5 * 5, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.max_pool2d(x, 2, 2)
        x = torch.relu(self.conv2(x))
        x = torch.max_pool2d(x, 2, 2)
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return torch.log_softmax(x, dim=1)

if __name__ == "__main__":
    # Load the existing model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = 83  # Update based on the actual number of classes in your dataset
    model = MathSymbolCNN(num_classes=num_classes).to(device)

    # Define the transformations
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    # Load the dataset
    train_dataset = datasets.ImageFolder(root=extracted_images_dir, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    # Define the loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    epochs = 10
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, target)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

        accuracy = 100. * correct / total
        print(f'Epoch {epoch + 1}, Loss: {running_loss / len(train_loader)}, Accuracy: {accuracy}%')

    # Save the trained model
    torch.save(model.state_dict(), "math_symbol_recognition_model.pth")
    print("Model saved.")