import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score

from src.ecg_dataset import PTBXLDataset
from src.model import ECGCNN


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASSES = ["NORM", "MI", "STTC", "CD", "HYP"]
THRESHOLD = 0.5


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
    print(f"Decision threshold: {THRESHOLD}")

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

    binary_preds = (all_preds >= THRESHOLD).astype(int)

    try:
        macro_auc = roc_auc_score(all_targets, all_preds, average="macro")
        print("\nMacro ROC-AUC:", round(macro_auc, 4))
    except Exception:
        print("\nMacro ROC-AUC: not computable")

    print("\nPer-class metrics:")
    for i, cls in enumerate(CLASSES):
        y_true = all_targets[:, i]
        y_prob = all_preds[:, i]
        y_pred = binary_preds[:, i]

        prevalence = float(y_true.mean())

        try:
            auc = roc_auc_score(y_true, y_prob)
            auc_text = f"{auc:.4f}"
        except Exception:
            auc_text = "not computable"

        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)

        print(
            f"{cls}: "
            f"prevalence={prevalence:.3f}, "
            f"AUC={auc_text}, "
            f"precision={precision:.3f}, "
            f"recall={recall:.3f}, "
            f"f1={f1:.3f}"
        )


if __name__ == "__main__":
    main()