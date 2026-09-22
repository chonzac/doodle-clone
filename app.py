import time
import tkinter as tk
from PIL import Image, ImageDraw, ImageOps
import numpy as np
import torch
import torch.nn as nn

# same as the training file
class SmallCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.features(x).flatten(1)
        return self.classifier(x)

def load_model(path="doodle_model.pt", device="cpu"):
    ckpt = torch.load(path, map_location=device)
    classes = ckpt["classes"]
    model = SmallCNN(len(classes)).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, classes

def preprocess_canvas(pil_img: Image.Image):
    """
    Canvas is black-on-white. QuickDraw bitmaps behave like white-on-black,
    so we invert to match.
    """
    img = pil_img.convert("L")
    img = ImageOps.invert(img)
    img = img.resize((28, 28), Image.Resampling.BILINEAR)
    arr = np.array(img, dtype=np.float32) / 255.0
    x = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)  # (1,1,28,28)
    return x

@torch.no_grad()
def predict(model, classes, pil_img, device="cpu", topk=3):
    x = preprocess_canvas(pil_img).to(device)
    logits = model(x)
    probs = torch.softmax(logits, dim=1).squeeze(0)
    vals, idxs = torch.topk(probs, k=min(topk, len(classes)))
    return [(classes[i], float(v)) for v, i in zip(vals, idxs)]

class DoodleApp:
    def __init__(self, root, model_path="doodle_model.pt", idle_seconds=1.0, brush=10):
        self.root = root
        self.root.title("Doodle AI (cat/dog/house/...)")

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, self.classes = load_model(model_path, device=self.device)

        self.canvas_size = 320
        self.brush = brush
        self.idle_seconds = idle_seconds

        self.last_draw_time = 0.0
        self.last_prediction_time = 0.0
        self.prev = None

        self.canvas = tk.Canvas(root, width=self.canvas_size, height=self.canvas_size, bg="white")
        self.canvas.grid(row=0, column=0, columnspan=3, padx=10, pady=10)

        self.pred_var = tk.StringVar(value="Draw something… (predicts after you pause)")
        tk.Label(root, textvariable=self.pred_var, font=("Arial", 14)).grid(row=1, column=0, columnspan=3, pady=(0, 10))

        tk.Button(root, text="Clear", command=self.clear).grid(row=2, column=0, sticky="ew", padx=10, pady=10)
        tk.Label(root, text=f"Device: {self.device}").grid(row=2, column=1)
        tk.Label(root, text=f"Idle: {idle_seconds:.1f}s").grid(row=2, column=2, padx=10)

        # Offscreen canvas image (so we can classify reliably)
        self.offscreen = Image.new("RGB", (self.canvas_size, self.canvas_size), "white")
        self.drawer = ImageDraw.Draw(self.offscreen)

        self.canvas.bind("<Button-1>", self.on_down)
        self.canvas.bind("<B1-Motion>", self.on_move)
        self.canvas.bind("<ButtonRelease-1>", self.on_up)

        self.root.after(200, self.loop)

    def on_down(self, e):
        self.prev = (e.x, e.y)
        self.last_draw_time = time.time()

    def on_move(self, e):
        x, y = e.x, e.y
        if self.prev is None:
            self.prev = (x, y)
            return

        x0, y0 = self.prev
        self.canvas.create_line(x0, y0, x, y, width=self.brush, fill="black",
                                capstyle=tk.ROUND, smooth=True)

        self.drawer.line([x0, y0, x, y], fill="black", width=self.brush)

        self.prev = (x, y)
        self.last_draw_time = time.time()

    def on_up(self, e):
        self.prev = None
        self.last_draw_time = time.time()

    def clear(self):
        self.canvas.delete("all")
        self.offscreen = Image.new("RGB", (self.canvas_size, self.canvas_size), "white")
        self.drawer = ImageDraw.Draw(self.offscreen)
        self.pred_var.set("Draw something… (predicts after you pause)")
        self.last_draw_time = 0.0
        self.last_prediction_time = 0.0

    def loop(self):
        now = time.time()
        if self.last_draw_time > 0:
            idle_for = now - self.last_draw_time
            if idle_for >= self.idle_seconds and (now - self.last_prediction_time) > 0.5:
                preds = predict(self.model, self.classes, self.offscreen, device=self.device, topk=3)
                self.pred_var.set("  |  ".join([f"{name}: {p:.2f}" for name, p in preds]))
                self.last_prediction_time = now

        self.root.after(200, self.loop)

def main():
    root = tk.Tk()
    app = DoodleApp(root, model_path="doodle_model.pt", idle_seconds=1.0, brush=10)
    root.mainloop()

if __name__ == "__main__":
    main()
