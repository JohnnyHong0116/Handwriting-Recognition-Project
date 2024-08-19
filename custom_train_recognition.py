import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, SubsetRandomSampler
from torchvision import datasets, transforms
from sklearn.model_selection import train_test_split

# Define the Custom CNN model for digit recognition
class CustomDigitCNN(nn.Module):
    def __init__(self):
        super(CustomDigitCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.fc1 = nn.Linear(64 * 5 * 5, 128)  # Adjust input size after convolution and pooling
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.max_pool2d(x, 2, 2)
        x = torch.relu(self.conv2(x))
        x = torch.max_pool2d(x, 2, 2)
        x = x.view(x.size(0), -1)  # Flatten the tensor for the fully connected layer
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return torch.log_softmax(x, dim=1)

# Function to limit GPU memory usage
def limit_gpu_memory_usage(memory_fraction=0.5):
    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(memory_fraction)
        torch.cuda.empty_cache()

# Function to train the model and save it
def train_and_save_model(model, device, train_loader, val_loader, epochs=10):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    model.train()
    for epoch in range(epochs):
        correct = 0
        total = 0
        running_loss = 0.0

        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            # Calculate accuracy
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
            running_loss += loss.item()

            if batch_idx % 100 == 0:
                print(f'Train Epoch: {epoch} [{batch_idx*len(data)}/{len(train_loader.dataset)} ({100.*batch_idx/len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')

        # Update learning rate
        scheduler.step()

        # Calculate training accuracy and loss
        accuracy = 100. * correct / total
        avg_loss = running_loss / len(train_loader)
        print(f'Epoch {epoch}: Train Accuracy: {accuracy:.2f}% | Train Loss: {avg_loss:.6f}')

        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        val_running_loss = 0.0
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                loss = criterion(output, target)
                val_running_loss += loss.item()

                _, predicted = torch.max(output.data, 1)
                val_total += target.size(0)
                val_correct += (predicted == target).sum().item()

        val_accuracy = 100. * val_correct / val_total
        val_avg_loss = val_running_loss / len(val_loader)
        print(f'Epoch {epoch}: Validation Accuracy: {val_accuracy:.2f}% | Validation Loss: {val_avg_loss:.6f}')

    # Save the trained model
    torch.save(model.state_dict(), "custom_digit_recognition_model.pth")
    print("Model saved")

if __name__ == "__main__":
    # Limit GPU memory usage
    limit_gpu_memory_usage(memory_fraction=0.5)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Define transformations for data augmentation and normalization
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((28, 28)),
        transforms.RandomRotation(10),  # Data augmentation
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # Load the dataset
    dataset_dir = './ML_Dataset/number_dataset'
    dataset = datasets.ImageFolder(root=dataset_dir, transform=transform)

    # Split the dataset into training and validation sets
    train_idx, val_idx = train_test_split(list(range(len(dataset))), test_size=0.2, random_state=42)
    train_sampler = SubsetRandomSampler(train_idx)
    val_sampler = SubsetRandomSampler(val_idx)

    train_loader = DataLoader(dataset, batch_size=64, sampler=train_sampler)
    val_loader = DataLoader(dataset, batch_size=64, sampler=val_sampler)
    
    # Initialize the model and start training
    model = CustomDigitCNN().to(device)
    train_and_save_model(model, device, train_loader, val_loader, epochs=10)
