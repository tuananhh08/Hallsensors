import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader 
from sklearn.model_selection import train_test_split

from fcn import FCN
from loss import WeightedMSELoss

DATA_PATH = "B_norm.csv"
BATCH_SIZE = 64
EPOCHS = 65
LR = 1e-3 
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class InverseDataset(Dataset):
    def __init__(self,X,y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
    
def train():
    data = np.load(DATA_PATH)
    X = data["B"].reshape(-1, 1, 8, 8)
    y = data["pose"]

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=True
    )

    train_loader = DataLoader(
        InverseDataset(X_train, y_train),
        batch_size=BATCH_SIZE,
        shuffle=True
    )
    val_loader = DataLoader(
        InverseDataset(X_val, y_val),
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    model = FCN(out_dim=5).to(DEVICE)
    criterion = WeightedMSELoss(w_xyz=1.0, w_angle=0.1)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    best_val_loss = float("inf")

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0

        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)

            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * xb.size(0)

        train_loss /= len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                loss = criterion(model(xb), yb)
                val_loss += loss.item() * xb.size(0)

        val_loss /= len(val_loader.dataset)

        print(
            f"Epoch [{epoch+1}/{EPOCHS}] "
            f"| Train {train_loss:.6f} "
            f"| Val {val_loss:.6f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                {"model_state_dict": model.state_dict()},
                "best_model.pth"
            )

    print("Training finished.")
    return model

def load_model(ckpt_path= "best_model.pth"):
    model = FCN(out_dim=5).to(DEVICE)
    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model

if __name__ == "__main__":
    train()