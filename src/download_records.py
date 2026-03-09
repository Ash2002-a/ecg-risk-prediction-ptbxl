import os
import urllib.request
import pandas as pd

DATA_PATH = "data/raw/ptb-xl"


def download_record(file_path):
    base_url = "https://physionet.org/files/ptb-xl/1.0.3/"
    url_base = base_url + file_path

    local_base = os.path.join(DATA_PATH, file_path)
    os.makedirs(os.path.dirname(local_base), exist_ok=True)

    for ext in [".hea", ".dat"]:
        url = url_base + ext
        out = local_base + ext

        if not os.path.exists(out):
            urllib.request.urlretrieve(url, out)
            print("Downloaded:", out)


def main():

    df = pd.read_csv(os.path.join(DATA_PATH, "ptbxl_database.csv"))

    # download first 200 ECGs
    for i in range(200):
        file_path = df.iloc[i]["filename_lr"]
        download_record(file_path)


if __name__ == "__main__":
    main()