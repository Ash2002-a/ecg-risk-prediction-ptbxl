import os
import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import DataLoader

from src.ecg_dataset import PTBXLDataset
from src.model import ECGCNN


BATCH_SIZE = 8
EPOCHS = 3
LEARNING_RATE = 1e-3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    df = pd.read_csv("data/raw/ptb-xl/ptbxl_database.csv")

    train_df = df[df["strat_fold"] <= 8].copy()
    val_df = df[df["strat_fold"] == 9].copy()

    train_dataset = PTBXLDataset(df=train_df)
    val_dataset = PTBXLDataset(df=val_df)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = ECGCNN(num_classes=5).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print(f"Using device: {DEVICE}")
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0

        for x, y in train_loader:
            x = x.to(DEVICE)
            y = y.to(DEVICE)

            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)

        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(DEVICE)
                y = y.to(DEVICE)

                logits = model(x)
                loss = criterion(logits, y)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)

        print(
            f"Epoch [{epoch + 1}/{EPOCHS}] "
            f"Train Loss: {avg_train_loss:.4f} "
            f"Val Loss: {avg_val_loss:.4f}"
        )

    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), "models/ecg_cnn.pt")
    print("Saved model to models/ecg_cnn.pt")


if __name__ == "__main__":
    main()