import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score

from src.ecg_dataset import PTBXLDataset
from src.model import ECGCNN


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASSES = ["NORM", "MI", "STTC", "CD", "HYP"]


def main():
    df = pd.read_csv("data/raw/ptb-xl/ptbxl_database.csv")
    test_df = df[df["strat_fold"] == 10].copy()

    test_dataset = PTBXLDataset(df=test_df)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

    model = ECGCNN(num_classes=len(CLASSES))
    model.load_state_dict(torch.load("models/ecg_cnn.pt", map_location=DEVICE))
    model.to(DEVICE)
    model.eval()

    print(f"Using device: {DEVICE}")
    print(f"Test samples: {len(test_dataset)}")

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(DEVICE)

            logits = model(x)
            probs = torch.sigmoid(logits)

            all_preds.append(probs.cpu().numpy())
            all_targets.append(y.numpy())

    all_preds = np.vstack(all_preds)
    all_targets = np.vstack(all_targets)

    try:
        macro_auc = roc_auc_score(all_targets, all_preds, average="macro")
        print("Macro ROC-AUC:", round(macro_auc, 4))
    except Exception:
        print("Macro ROC-AUC: not computable")

    for i, cls in enumerate(CLASSES):
        try:
            auc = roc_auc_score(all_targets[:, i], all_preds[:, i])
            print(f"{cls} AUC:", round(auc, 4))
        except Exception:
            print(f"{cls} AUC: not computable")


if __name__ == "__main__":
    main()