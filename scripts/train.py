import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data import get_dataloaders
from src.model import SimpsonClassifier
from scripts.val import validate

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.transforms import transforms

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt


def train_model(model, train_loader, val_loader, epochs):

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    losses = []

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        all_preds, all_labels = [], []
        for img, label in train_loader:
            optimizer.zero_grad()
            output = model(img)
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            all_preds.extend(predicted.numpy())
            all_labels.extend(label.numpy())
        print(f"Epoch {epoch + 1}/{epochs}")


        train_loss = running_loss / len(train_loader)
        losses.append(train_loss)
            
        train_acc = accuracy_score(all_labels, all_preds)
        train_prec = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
        train_rec = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
        train_f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)

        print(f"Train Loss:      {train_loss:.4f}")
        print(f"Train Accuracy:  {train_acc:.4f}")
        print(f"Train Precision: {train_prec:.4f}")
        print(f"Train Recall:    {train_rec:.4f}")
        print(f"Train F1-Score:  {train_f1:.4f}")


    torch.save(model.state_dict(), 'model.pth')

    build_graph_loss(losses)


def build_graph_loss(losses):
    plt.figure(figsize=(8, 4))
    plt.plot(losses, label='Train Loss', color='red', linewidth=2)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('loss_graph.png', dpi=150)


def main():
    root_dir = "./data/simpsons_dataset"
    train_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    val_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    train_loader, _ = get_dataloaders(root_dir, train_transform)
    _, val_loader = get_dataloaders(root_dir, val_transform)

    model = SimpsonClassifier()
    train_model(model, train_loader, val_loader, 10)


if __name__ == "__main__":
    main()
