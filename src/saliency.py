import torch
import matplotlib.pyplot as plt

from src.ecg_dataset import PTBXLDataset
from src.model import ECGCNN

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def compute_saliency(model, x):
    x = x.unsqueeze(0).to(DEVICE)
    x.requires_grad_()

    logits = model(x)
    score = logits.max()

    score.backward()

    saliency = x.grad.abs().cpu()[0]

    return saliency


def main():

    dataset = PTBXLDataset(max_records=200)

    x, y = dataset[5]

    model = ECGCNN(num_classes=5)
    model.load_state_dict(torch.load("models/ecg_cnn.pt", map_location=DEVICE))
    model.to(DEVICE)
    model.eval()

    saliency = compute_saliency(model, x)

    signal = x[0].numpy()
    sal = saliency[0].numpy()

    plt.figure(figsize=(12,4))
    plt.plot(signal, label="ECG")
    plt.plot(sal / sal.max(), label="Saliency", alpha=0.7)
    plt.legend()
    plt.title("ECG Saliency Map (Lead I)")
    plt.xlabel("Sample")
    plt.ylabel("Amplitude")

    plt.tight_layout()

    import os
    os.makedirs("outputs/figures", exist_ok=True)
    plt.savefig("outputs/figures/saliency_map.png", dpi=300)

    plt.show()


if __name__ == "__main__":
    main()