# train.py
import torch
import torch.nn as nn
from models.cnn_model import SimpleCNN
from utils.dataloader import get_dataloader
from config import config

def train():
    # Check if GPU is available and set the device accordingly
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SimpleCNN(num_classes=config["num_classes"]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    criterion = nn.CrossEntropyLoss()
    train_loader = get_dataloader(config["data_path"], config["batch_size"])

    for epoch in range(config["epochs"]):
        model.train()
        total_loss = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch [{epoch+1}/{config['epochs']}], Loss: {total_loss / len(train_loader)}")

    # Save the model
    torch.save(model.state_dict(), config["model_save_path"])
