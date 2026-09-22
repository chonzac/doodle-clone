import torch
import torch.nn as nn
import numpy as np
from PIL import Image, ImageOps


class SmallCNN(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(),
            nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(),

            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.classifier = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.features(x).flatten(1)
        return self.classifier(x)



def load_checkpoint(path="doodle_model.pt"):
    ckpt = torch.load(path, map_location="cpu")
    classes = ckpt["classes"]
    model = SmallCNN(len(classes))
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, classes


def preprocess_pil(pil_img: Image.Image, invert=True) -> torch.Tensor:
    import numpy as np
    from PIL import ImageOps, Image

    img = pil_img.convert("L")
    if invert:
        img = ImageOps.invert(img)

    arr = np.array(img, dtype=np.uint8)

    # Find "ink" pixels (anything not near black background)
    ys, xs = np.where(arr > 30)  # threshold can be tuned (20-50)
    if len(xs) > 0 and len(ys) > 0:
        x0, x1 = xs.min(), xs.max()
        y0, y1 = ys.min(), ys.max()

        # Pad the bounding box a bit
        pad = 10
        x0 = max(0, x0 - pad); y0 = max(0, y0 - pad)
        x1 = min(arr.shape[1] - 1, x1 + pad); y1 = min(arr.shape[0] - 1, y1 + pad)

        img = img.crop((x0, y0, x1 + 1, y1 + 1))

    # Make square (centered) to avoid distortion
    w, h = img.size
    side = max(w, h)
    square = Image.new("L", (side, side), color=0)  # background black (since inverted)
    square.paste(img, ((side - w) // 2, (side - h) // 2))

    # Resize to model input
    square = square.resize((28, 28), Image.Resampling.BILINEAR)

    arr2 = np.array(square, dtype=np.float32) / 255.0
    x = torch.from_numpy(arr2).unsqueeze(0).unsqueeze(0)  # (1,1,28,28)
    return x



@torch.no_grad()
def predict(model, classes, pil_img: Image.Image, invert=True, topk=1):
    x = preprocess_pil(pil_img, invert=invert)
    logits = model(x)
    probs = torch.softmax(logits, dim=1).squeeze(0)
    vals, idxs = torch.topk(probs, k=min(topk, len(classes)))
    out = []
    for v, i in zip(vals.tolist(), idxs.tolist()):
        out.append((classes[i], float(v)))
    return out
