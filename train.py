import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split

CLASSES = [
    "apple",
    "banana",
    "calculator",
    "car",
    "cat",
    "chair",
    "clock",
    "cup",
    "dog",
    "door",
    "face",
    "fork",
    "house",
    "key",
    "ladder",
    "scissors",
    "television",
    "train",
    "tree",
    "umbrella",
    "telephone",
    "The Eiffel Tower",
    "alarm clock",
    "angel",
    "axe",
    "baseball bat",
    "basketball",
    "bee",
    "bicycle",
    "boomerang",
    "brain",
    "bread",
    "bus",
    "bush",
    "cactus",
    "camel",
    "carrot",
    "cloud",
    "computer",
    "cookie",
    "cow",
    "crab",
    "crayon",
    "crown",
    "diamond",
    "donut",
    "eye",
    "fire hydrant",
    "fireplace",
    "fish",
    "flower",
    "flying saucer",
    "giraffe",
    "hammer",
    "horse",
    "hospital",
    "hot air balloon",
    "hot dog",
    "hourglass",
    "knife",
    "light bulb",
    "microphone",
    "monkey",
    "moon",
    "mountain",
    "mouse",
    "mushroom",
    "palm tree",
    "penguin",
    "pineapple",
    "shark",

]

class QuickDrawNPY(Dataset):
    """
    Loads multiple QuickDraw .npy files (each: N x 784 uint8),
    samples up to per_class per category, returns (1,28,28) tensor in [0,1].
    """
    def __init__(self, npy_dir: str, classes, per_class=20000, seed=123):
        rng = np.random.default_rng(seed)

        xs = []
        ys = []
        for i, name in enumerate(classes):
            path = os.path.join(npy_dir, f"{name}.npy")
            if not os.path.exists(path):
                raise FileNotFoundError(f"Missing {path}. Run download_data.py first.")

            arr = np.load(path)  # (N, 784)
            n = min(per_class, arr.shape[0])

            # random sample for better variety
            idx = rng.choice(arr.shape[0], size=n, replace=False)
            arr = arr[idx]

            xs.append(arr)
            ys.append(np.full((n,), i, dtype=np.int64))

        x = np.concatenate(xs, axis=0)
        y = np.concatenate(ys, axis=0)

        # shuffle whole dataset once
        perm = rng.permutation(len(y))
        self.x = x[perm]
        self.y = y[perm]
        self.classes = classes

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        img = self.x[idx].reshape(28, 28).astype(np.float32) / 255.0
        img = torch.from_numpy(img).unsqueeze(0)  # (1,28,28)
        label = torch.tensor(self.y[idx], dtype=torch.long)
        return img, label

class SmallCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),

            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(),

            nn.AdaptiveAvgPool2d((1, 1))
        )

        self.classifier = nn.Linear(256, num_classes)


    def forward(self, x):
        x = self.features(x).flatten(1)
        return self.classifier(x)

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct = total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        pred = model(x).argmax(1)
        correct += (pred == y).sum().item()
        total += y.numel()
    return correct / max(total, 1)

def main(
    npy_dir="quickdraw_npy",
    per_class=20000,
    batch_size=256,
    epochs=15,
    lr=1e-3,
    val_split=0.10,
    out_path="doodle_model.pt"
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    ds = QuickDrawNPY(npy_dir, CLASSES, per_class=per_class)
    val_size = int(len(ds) * val_split)
    train_size = len(ds) - val_size
    train_ds, val_ds = random_split(ds, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    model = SmallCNN(num_classes=len(CLASSES)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    best = 0.0
    for epoch in range(1, epochs + 1):
        model.train()
        running = 0.0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            logits = model(x)
            loss = loss_fn(logits, y)
            loss.backward()
            opt.step()
            running += loss.item() * y.size(0)

        val_acc = evaluate(model, val_loader, device)
        train_loss = running / train_size
        print(f"epoch {epoch:02d} | train_loss={train_loss:.4f} | val_acc={val_acc:.4f}")

        if val_acc > best:
            best = val_acc
            torch.save({"model": model.state_dict(), "classes": CLASSES}, out_path)
            print(f"  saved {out_path} (best val_acc={best:.4f})")

    print("done. best val_acc:", best)

if __name__ == "__main__":
    main()
