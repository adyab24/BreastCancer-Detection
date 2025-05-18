import os
import pandas as pd
from PIL import Image
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import Dataset as TorchDataset, DataLoader
from torchvision import transforms
from efficientnet_pytorch import EfficientNet

# --- 1. Direct Dataset Loader from DataFrame ---
class Dataset(TorchDataset):
    def __init__(self, df, img_dir, transform=None):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.img_dir, row["image_name"])
        image = Image.open(img_path).convert("RGB")
        label = int(row["birads_class"])

        if self.transform:
            image = self.transform(image)
        return image, label

# --- 2. Classifier Model ---
class LinearClassifier(nn.Module):
    def __init__(self, feature_dim, num_classes):
        super().__init__()
        self.classification_head = nn.Linear(feature_dim, num_classes)

    def forward(self, x):
        return self.classification_head(x)

class classifier(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()
        self.image_encoder = EfficientNet.from_name("efficientnet-b2")
        self.out_dim = self.image_encoder._fc.in_features
        self.image_encoder._fc = nn.Identity()
        self.classifier = LinearClassifier(self.out_dim, num_classes)

    def forward(self, x):
        features = self.image_encoder(x)
        return self.classifier(features)

# --- 3. Training Function ---
def train(model, loader, optimizer, criterion, device):
    model.train()
    total, correct, total_loss = 0, 0, 0
    for imgs, labels in tqdm(loader, desc="Training"):
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * imgs.size(0)
        preds = outputs.argmax(1)
        correct += (preds == labels).sum().item()
        total += imgs.size(0)

    return total_loss / total, correct / total

# --- 4. Validation Function ---
def evaluate(model, loader, criterion, device):
    model.eval()
    total, correct, total_loss = 0, 0, 0
    with torch.no_grad():
        for imgs, labels in tqdm(loader, desc="Validating"):
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * imgs.size(0)
            preds = outputs.argmax(1)
            correct += (preds == labels).sum().item()
            total += imgs.size(0)

    return total_loss / total, correct / total

# --- 5. Run Training ---
def main():
    # --- Config ---
    annotation_csv_path = "breast_level_annotations.csv"  # full CSV
    img_dir = "images/"                                   # .jpg images
    image_ext = ".jpg"
    save_path = "model_weights.pth"
    batch_size = 16
    epochs = 20
    lr = 1e-3
    val_split_ratio = 0.2

    # --- Load and Process CSV ---
    full_df = pd.read_csv(annotation_csv_path)
    full_df = full_df[full_df["split"].isin(["train", "val"])].copy()

    def map_birads(b):
        if b in [1, 2]: return 0
        elif b in [3, 4]: return 1
        elif b == 5: return 2
        else: return None

    full_df["birads_class"] = full_df["breast_birads"].apply(map_birads)
    full_df["image_name"] = full_df["image_id"].astype(str) + image_ext
    full_df = full_df.dropna(subset=["birads_class"])

    # --- Split ---
    val_size = int(len(full_df) * val_split_ratio)
    train_df = full_df.sample(frac=1 - val_split_ratio, random_state=42)
    val_df = full_df.drop(train_df.index)

    # --- Transforms ---
    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])

    # --- Dataloaders ---
    train_loader = DataLoader(Dataset(train_df, img_dir, transform), batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(Dataset(val_df, img_dir, transform), batch_size=batch_size, shuffle=False, num_workers=4)

    # --- Model Setup ---
    device = torch.device("mps")
    model = classifier(num_classes=3).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    # --- Training Loop ---
    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        print(f"Epoch {epoch:2d} | Loss: {train_loss:.4f} | Train Acc: {train_acc:.3f} | Val Acc: {val_acc:.3f}")
        torch.save(model.state_dict(), save_path)

if __name__ == "__main__":
    main()
