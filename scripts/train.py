import sys
import os
import random
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from torchvision.transforms import transforms

from src.data import get_dataloaders
from src.model import SimpsonClassifier
from scripts.val import validate


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)


def train_model(model, train_loader, val_loader, epochs):

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    losses = []

    best_val_acc = 0
    early_stopping = 0

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

        val_acc, val_prec, val_rec,  val_f1 = validate(model, val_loader)

        print(f"Val Accuracy:  {val_acc:.4f}")
        print(f"Val Precision: {val_prec:.4f}")
        print(f"Val Recall:    {val_rec:.4f}")
        print(f"Val F1-Score:  {val_f1:.4f}")

        if (best_val_acc < val_acc):
            best_val_acc = val_acc
            early_stopping = 0
            torch.save(model.state_dict(), 'best_model.pth')
        else:
            early_stopping += 1

            if (early_stopping > 5):
                print(f"EARLY STOPPING - Epoch {epoch + 1}/{epochs}")
                break

    torch.save(model.state_dict(), 'last_model.pth')

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
    set_seed(42)

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
    train_model(model, train_loader, val_loader, 30)


if __name__ == "__main__":
    main()
