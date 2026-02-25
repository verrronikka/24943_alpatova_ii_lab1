import os
import cv2
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split


class SimpsonDataset(Dataset):
    def __init__(self, imgs, labels, transform=None):

        self.imgs = imgs
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):

        img_path = self.imgs[idx]

        img = cv2.imread(img_path, cv2.IMREAD_COLOR_RGB)
        input_img = self.transform(img)

        return input_img, self.labels[idx]


def get_dataloaders(root_dir, transform):
    images, labels = [], []
    classes = sorted([d for d in os.listdir(root_dir)])

    class_idx = {name: idx for idx, name in enumerate(classes)}

    for name, idx in class_idx.items():
        folder = os.path.join(root_dir, name)
        for img in os.listdir(folder):
            img_path = os.path.join(folder, img)
            if os.path.isfile(img_path) and img.lower().endswith(('.jpg', '.jpeg')):
                images.append(img_path)
                labels.append(idx)

    X_train, x_val, y_train, y_val = train_test_split(
        images, labels, test_size=0.2, stratify=labels, random_state=42
    )

    train_dataset = SimpsonDataset(X_train, y_train, transform)
    train_loader = DataLoader(
        train_dataset, batch_size=32, shuffle=True, num_workers=4)

    val_dataset = SimpsonDataset(x_val, y_val, transform)
    val_loader = DataLoader(
        val_dataset, batch_size=32, shuffle=True, num_workers=4)

    return train_loader, val_loader
