import argparse
import csv
import pickle

import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from tqdm import tqdm

TEST_FILE = "data/cifar_test_nolabels.pkl"

# Get experiment id and model path
parser = argparse.ArgumentParser()
parser.add_argument("--id", type=str, required=True)
args = parser.parse_args()
model_path = f"logs/{args.id}/ckpt.pth"


# Prepare test data
transform_test = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ]
)


class TestDataset(torch.utils.data.Dataset):
    def __init__(self, file, transform):
        self.raw_data = self.load_data(file)
        self.transform = transform

    def load_data(self, file: str) -> dict:
        """
        Read raw test data from a pickle file.

        Args:
            file (str): The path to the test file.

        Returns:
            dict: The dictionary containing the following:
                b"data": a 10000x3072 numpy array of uint8s. Each row of the array stores a 32x32 colour image.
                    The first 1024 entries contain the red channel values, the next 1024 the green, and the final 1024 the blue.
                    The image is stored in row-major order, so that the first 32 entries of the array are the red channel values of the first row of the image.
                b"ids": a list of 10000 numbers in the range 0-9. The number at index i indicates the label of the ith image in the array data.
        """
        with open(file, "rb") as fo:
            raw_data = pickle.load(fo, encoding="bytes")
        return raw_data

    def decode_image(self, raw_image: np.ndarray) -> Image:
        """
        Decode a single image from raw image.

        Args:
            raw_image (np.ndarray): A (3072,) numpy array of uint8s. Each row of the array stores a 32x32 colour image.
                The first 1024 entries contain the red channel values, the next 1024 the green, and the final 1024 the blue.
                The image is stored in row-major order, so that the first 32 entries of the array are the red channel values of the first row of the image.

        Returns:
            Image: PIL Image object with shape (32, 32) and mode "RGB".
        """
        image = raw_image.reshape(3, 32, 32)
        image = image.transpose((1, 2, 0))
        image = Image.fromarray(image)
        return image

    def __len__(self):
        return len(self.raw_data[b"ids"])

    def __getitem__(self, idx):
        raw_image = self.raw_data[b"data"][idx]
        image = self.decode_image(raw_image)
        image = self.transform(image)

        ID = self.raw_data[b"ids"][idx]
        return image, ID


testset = TestDataset(TEST_FILE, transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False)

# Prepare model
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"
net = torch.jit.load(model_path)
net = net.to(device)
net = net.eval()

# Inference
predictions = {"ID": [], "Labels": []}
with torch.no_grad():
    for inputs, IDs in tqdm(testloader, dynamic_ncols=True):
        inputs = inputs.to(device)
        outputs = net(inputs)
        _, predicted = outputs.max(1)
        predictions["ID"].extend(IDs.cpu().tolist())
        predictions["Labels"].extend(predicted.cpu().tolist())

# Save predictions
with open("predictions.csv", "w", newline="") as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=predictions.keys())
    writer.writeheader()
    for i, label in zip(predictions["ID"], predictions["Labels"]):
        writer.writerow({"ID": i, "Labels": label})
