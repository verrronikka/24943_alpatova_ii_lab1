import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data import get_test_loaders
from src.model import SimpsonClassifier
from scripts.val import validate

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.transforms import transforms



def main():
    train_dir = "./data/simpsons_dataset"
    test_dir = "./data/kaggle_simpson_testset/"

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    test_loader = get_test_loaders(train_dir, test_dir, transform)

    model = SimpsonClassifier()
    model.load_state_dict(torch.load("best_model.pth"))
    acc, prec, rec, f1 = validate(model, test_loader)

    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1-Score:  {f1:.4f}")


if __name__=="__main__":
    main()
