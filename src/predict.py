import os
import torch
import matplotlib.pyplot as plt

from src.ecg_dataset import PTBXLDataset, TARGET_CLASSES
from src.model import ECGCNN


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    dataset = PTBXLDataset(max_records=200)

    x, y = dataset[5]

    model = ECGCNN(num_classes=len(TARGET_CLASSES))
    model.load_state_dict(torch.load("models/ecg_cnn.pt", map_location=DEVICE))
    model.to(DEVICE)
    model.eval()

    with torch.no_grad():
        logits = model(x.unsqueeze(0).to(DEVICE))
        probs = torch.sigmoid(logits).cpu().numpy()[0]

    print("\nPredicted probabilities:")
    for cls, p in zip(TARGET_CLASSES, probs):
        print(f"{cls}: {p:.3f}")

    true_labels = [cls for cls, v in zip(TARGET_CLASSES, y) if v == 1]
    print("\nTrue label(s):")
    for label in true_labels:
        print(label)

    os.makedirs("outputs/figures", exist_ok=True)

    signal = x[0].numpy()
    plt.figure(figsize=(12, 4))
    plt.plot(signal)
    plt.title("ECG Lead I with Model Prediction")
    plt.xlabel("Sample")
    plt.ylabel("Amplitude")
    plt.tight_layout()
    plt.savefig("outputs/figures/prediction_example.png", dpi=300)
    plt.show()

    print("\nSaved figure to outputs/figures/prediction_example.png")


if __name__ == "__main__":
    main()