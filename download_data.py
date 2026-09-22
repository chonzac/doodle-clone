import os
import urllib.request

BASE = "https://storage.googleapis.com/quickdraw_dataset/full/numpy_bitmap/"

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



def download(out_dir="quickdraw_npy"):
    os.makedirs(out_dir, exist_ok=True)
    for name in CLASSES:
        filename = f"{name}.npy"
        url = BASE + filename.replace(" ", "%20")
        dst = os.path.join(out_dir, filename)

        if os.path.exists(dst):
            print(f"exists: {dst}")
            continue

        print(f"downloading: {name}")
        urllib.request.urlretrieve(url, dst)
        print(f"saved: {dst}")

if __name__ == "__main__":
    download()
