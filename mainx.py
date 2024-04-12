import argparse
import json
import os
import shutil
import torch.nn.init as init

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm

# Directories to save logs and data
LOG_DIR = "./logs"
DATA_DIR = "./data"

# Check if dev mode
# In dev mode, only the first 100 training and testing data will be used and 1 epoch will be run.
# The log will be saved in logs/dev.
parser = argparse.ArgumentParser()
parser.add_argument("--dev", action="store_true")
parser.add_argument("--ks", type=int, default=1)
parser.add_argument("--sks", type=int, default=1)
parser.add_argument("--ch", type=int, default=64)

args = parser.parse_args()
dev = args.dev

if dev:
    print("Running in dev mode.")


# Log experiment
# New experiment id is created by counting the number of existing experiments.
i = 0
while os.path.exists(os.path.join(LOG_DIR, str(i))):
    i += 1
experiment_id = str(i) if not dev else "dev"
experiment_log = os.path.join(LOG_DIR, experiment_id)
os.makedirs(experiment_log, exist_ok=True)
shutil.copy(__file__, experiment_log)
print(f"Starting experiment {experiment_id}.")

# Set random seed for reproducibility
torch.manual_seed(42)

# Data
transform_train = transforms.Compose(
    [
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
        transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ]
)

transform_test = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ]
)

trainset = torchvision.datasets.CIFAR10(
    root=DATA_DIR, train=True, download=True, transform=transform_train
)
testset = torchvision.datasets.CIFAR10(
    root=DATA_DIR, train=False, download=True, transform=transform_test
)
if dev:
    trainset = torch.utils.data.Subset(trainset, list(range(100)))
    testset = torch.utils.data.Subset(testset, list(range(100)))

trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=64, shuffle=True, num_workers=4
)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=100, shuffle=False, num_workers=4
)

# # Save an example of transformed image
# images, _ = next(iter(trainloader))
# torchvision.utils.save_image(images[0], "transformed_image.png")


# Model
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, kernel_size=3, shortcut_kernel_size=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes,
                    self.expansion * planes,
                    kernel_size=shortcut_kernel_size,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(self.expansion * planes),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


scaler = torch.cuda.amp.GradScaler()

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_channels=64, num_classes=10, kernel_size=3, shortcut_kernel_size=1):
        super(ResNet, self).__init__()
        self.in_planes = num_channels
        self.kernel_size = kernel_size
        self.shortcut_kernel_size = shortcut_kernel_size
        self.conv1 = nn.Conv2d(3, num_channels, kernel_size=kernel_size, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(num_channels)

        # Residual layers
        layers = []
        for i, num_block in enumerate(num_blocks):
            stride = 1 if i == 0 else 2
            layers.append(self._make_layer(block, num_channels * 2**i, num_block, stride=stride))
        self.layers = nn.Sequential(*layers)

        # Average pooling
        self.avg_pool = nn.AvgPool2d(32 // (2 ** (len(num_blocks) - 1)))

        # FC
        self.linear = nn.Linear(
            num_channels * 2 ** (len(num_blocks) - 1) * block.expansion, num_classes
        )

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, kernel_size=self.kernel_size, shortcut_kernel_size=self.shortcut_kernel_size))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layers(out)
        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


net = ResNet(BasicBlock, [4, 4, 3], num_channels=args.ch, kernel_size=args.ks, shortcut_kernel_size=args.sks)

n_parameters = sum(p.numel() for p in net.parameters() if p.requires_grad)
print(f"Number of parameters: {n_parameters:,}")

# Training
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.05, momentum=0.9, weight_decay=5e-4)
# optimizer = optim.Adam(net.parameters(), lr=3e-4, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():  # For Apple silicon
    device = "mps"
else:
    device = "cpu"
net = net.to(device)



def initialize_weights(model):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            init.xavier_uniform_(m.weight)
            if m.bias is not None:
                init.zeros_(m.bias)

# initialize_weights(net)

# net.apply(lambda m: torch.nn.init.xavier_uniform_(m.weight.data))

def train(epoch):
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    with tqdm(trainloader, dynamic_ncols=True) as tepoch:
        tepoch.set_description(f"Training epoch: {epoch}")
        for batch_idx, (inputs, targets) in enumerate(tepoch):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()

            with torch.cuda.amp.autocast():
                outputs = net(inputs)
                loss = criterion(outputs, targets)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()


            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            tepoch.set_postfix(loss=train_loss / (batch_idx + 1), acc=correct / total)

    log = {
        "loss": train_loss / len(trainloader),
        "acc": correct / total,
    }
    return log


def test(epoch):
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        with tqdm(testloader, dynamic_ncols=True) as tepoch:
            tepoch.set_description(f"Testing  epoch: {epoch}")
            for batch_idx, (inputs, targets) in enumerate(tepoch):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = net(inputs)
                loss = criterion(outputs, targets)

                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

                tepoch.set_postfix(
                    loss=test_loss / (batch_idx + 1), acc=correct / total
                )

    log = {
        "net": net,
        "loss": test_loss / len(testloader),
        "acc": correct / total,
    }
    return log


best_acc = 0
best_epoch = 0
logs = {}
for epoch in range(200):
    train_log = train(epoch)
    test_log = test(epoch)
    scheduler.step()

    # Save checkpoint based on best testing accuracy
    if test_log["acc"] > best_acc:
        torch.jit.script(test_log["net"]).save(
            os.path.join(experiment_log, f"ckpt.pth")
        )
        best_acc = test_log["acc"]
        best_epoch = epoch

    # Log results
    logs["train_loss"] = logs.get("train_loss", []) + [train_log["loss"]]
    logs["train_acc"] = logs.get("train_acc", []) +      [train_log["acc"]]
    logs["test_loss"] = logs.get("test_loss", []) + [test_log["loss"]]
    logs["test_acc"] = logs.get("test_acc", []) + [test_log["acc"]]

    if dev:
        break

# Save logged results
with open(os.path.join(experiment_log, "logs.json"), "w") as f:
    json.dump(logs, f)

print(f"Experiment {experiment_id}: Best accuracy {best_acc} at epoch {best_epoch}")
