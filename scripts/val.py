import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data import get_dataloaders
from src.model import SimpsonClassifier

import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import torch.nn as nn
import torch.optim as optim
from torchvision.transforms import transforms

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def build_confusion_matrix(cm, class_names=None):
    plt.figure(figsize=(12, 10))

    if class_names is None:
        class_names = [f"Class {i}" for i in range(cm.shape[0])]
    

    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, 
                yticklabels=class_names,
                linewidths=0.5)
    

    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig("confusion_matrix.png", dpi=300)
    plt.close()


def validate(model, val_loader):
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for img, label in val_loader:
            output = model(img)
            _, predicted = torch.max(output.data, 1)
            all_preds.extend(predicted.numpy())
            all_labels.extend(label.numpy())


    acc = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
    rec = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    cm = confusion_matrix(all_labels, all_preds)
    build_confusion_matrix(cm)
    return acc, prec, rec, f1


def main():
    root_dir = "./data/simpsons_dataset"
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    _, val_loader = get_dataloaders(root_dir, transform)

    model = SimpsonClassifier()
    model.load_state_dict(torch.load("best_model.pth"))
    acc, prec, rec, f1 = validate(model, val_loader)

    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1-Score:  {f1:.4f}")


if __name__=="__main__":
    main()
