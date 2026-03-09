import os
import ast
import wfdb
import pandas as pd


DATA_PATH = "data/raw/ptb-xl"


def load_metadata():
    path = os.path.join(DATA_PATH, "ptbxl_database.csv")
    df = pd.read_csv(path)
    return df


def load_scp_statements():
    path = os.path.join(DATA_PATH, "scp_statements.csv")
    scp = pd.read_csv(path, index_col=0)
    return scp


def get_diagnostic_superclass_map(scp):
    diagnostic_scp = scp[scp["diagnostic"] == 1.0]
    diagnostic_map = diagnostic_scp["diagnostic_class"].to_dict()
    return diagnostic_map


def parse_diagnostic_superclasses(scp_codes_str, diagnostic_map):
    scp_codes = ast.literal_eval(scp_codes_str)
    labels = sorted(set(code_map for code, code_map in (
        (code, diagnostic_map[code]) for code in scp_codes if code in diagnostic_map
    )))
    return labels


def load_ecg(record_path):
    signal, meta = wfdb.rdsamp(record_path)
    return signal, meta


if __name__ == "__main__":
    df = load_metadata()
    scp = load_scp_statements()
    diagnostic_map = get_diagnostic_superclass_map(scp)

    print("Total ECG records:", len(df))

    row = df.iloc[0]
    path = os.path.join(DATA_PATH, row["filename_lr"])

    signal, meta = load_ecg(path)
    labels = parse_diagnostic_superclasses(row["scp_codes"], diagnostic_map)

    print("ECG ID:", row["ecg_id"])
    print("Signal shape:", signal.shape)
    print("Sampling rate:", meta["fs"])
    print("Leads:", meta["sig_name"])
    print("Diagnostic superclasses:", labels)