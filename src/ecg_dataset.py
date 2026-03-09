import ast
import os
import wfdb
import torch
import pandas as pd
from torch.utils.data import Dataset


class PTBXLDataset(Dataset):
    def __init__(self, df=None, max_records=None):
        self.base_path = "data/raw/ptb-xl"
        self.classes = ["NORM", "MI", "STTC", "CD", "HYP"]

        if df is None:
            df = pd.read_csv(f"{self.base_path}/ptbxl_database.csv")

        scp = pd.read_csv(f"{self.base_path}/scp_statements.csv", index_col=0)
        diagnostic_map = scp[scp["diagnostic"] == 1.0]["diagnostic_class"].to_dict()

        records = []

        for _, row in df.iterrows():
            try:
                path = row["filename_lr"]
                hea_path = os.path.join(self.base_path, path + ".hea")
                dat_path = os.path.join(self.base_path, path + ".dat")

                # only keep records that actually exist locally
                if not (os.path.exists(hea_path) and os.path.exists(dat_path)):
                    continue

                codes = ast.literal_eval(row["scp_codes"])
                labels = sorted(set(diagnostic_map[c] for c in codes if c in diagnostic_map))
                labels = [label for label in labels if label in self.classes]

                if len(labels) == 0:
                    continue

                records.append((path, labels))

            except Exception:
                continue

        if max_records is not None:
            records = records[:max_records]

        self.records = records

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        path, labels = self.records[idx]

        signal, _ = wfdb.rdsamp(f"{self.base_path}/{path}")
        x = torch.tensor(signal.T, dtype=torch.float32)

        y = torch.zeros(len(self.classes), dtype=torch.float32)
        for label in labels:
            y[self.classes.index(label)] = 1.0

        return x, y