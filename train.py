import torch
import torch.nn as nn
import torch.optim as optim
from dataset import get_loaders
from model import WasteClassifier


def train_model():
    DATA_DIR = "./dataset"
    EPOCHS = 15
    BATCH_SIZE = 32
    LEARNING_RATE = 0.001

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on device: {device}")

    train_loader, val_loader, classes = get_loaders(DATA_DIR, BATCH_SIZE)

    model = WasteClassifier(num_classes=len(classes)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()

        epoch_loss = running_loss / len(train_loader.dataset)
        train_acc = (correct_train / total_train) * 100

        model.eval()
        correct_val = 0
        total_val = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()

        val_acc = (correct_val / total_val) * 100
        print(
            f"Epoch [{epoch+1}/{EPOCHS}] - Loss: {epoch_loss:.4f} - Train Acc: {train_acc:.2f}% - Val Acc: {val_acc:.2f}%")

    torch.save({
        'model_state_dict': model.state_dict(),
        'classes': classes
    }, "model.pth")
    print("Model saved successfully as model.pth!")


if __name__ == "__main__":
    train_model()
