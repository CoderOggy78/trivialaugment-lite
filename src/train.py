import argparse
import os
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as T

from trivial_augment import TrivialAugment
from model import wrn_28_10, wrn_40_2

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)
CIFAR100_MEAN = (0.5071, 0.4865, 0.4409)
CIFAR100_STD = (0.2673, 0.2564, 0.2762)


class Cutout:
    def __init__(self, size=16):
        self.size = size

    def __call__(self, img):
        h, w = img.shape[1], img.shape[2]
        y = torch.randint(h, (1,)).item()
        x = torch.randint(w, (1,)).item()
        half = self.size // 2
        y1, y2 = max(0, y - half), min(h, y + half)
        x1, x2 = max(0, x - half), min(w, x + half)
        img[:, y1:y2, x1:x2] = 0.0
        return img


def build_loaders(dataset, data_dir, batch_size, workers):
    mean, std = (CIFAR100_MEAN, CIFAR100_STD) if dataset == "cifar100" else (CIFAR10_MEAN, CIFAR10_STD)
    train_tf = T.Compose([
        T.RandomCrop(32, padding=4, padding_mode="reflect"),
        T.RandomHorizontalFlip(),
        TrivialAugment(),
        T.ToTensor(),
        T.Normalize(mean, std),
        Cutout(16),
    ])
    test_tf = T.Compose([T.ToTensor(), T.Normalize(mean, std)])

    cls = torchvision.datasets.CIFAR100 if dataset == "cifar100" else torchvision.datasets.CIFAR10
    train_set = cls(root=data_dir, train=True, download=True, transform=train_tf)
    test_set = cls(root=data_dir, train=False, download=True, transform=test_tf)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                               num_workers=workers, pin_memory=True, drop_last=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False,
                              num_workers=workers, pin_memory=True)
    return train_loader, test_loader


def evaluate(model, loader, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            pred = model(x).argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    return correct / total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["cifar10", "cifar100"], default="cifar10")
    parser.add_argument("--model", choices=["wrn_28_10", "wrn_40_2"], default="wrn_40_2")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--data-dir", default="./data")
    parser.add_argument("--checkpoint-dir", default="./checkpoints")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    num_classes = 100 if args.dataset == "cifar100" else 10
    train_loader, test_loader = build_loaders(args.dataset, args.data_dir, args.batch_size, args.workers)

    model = (wrn_28_10 if args.model == "wrn_28_10" else wrn_40_2)(num_classes=num_classes).to(device)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9,
                           nesterov=True, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss()

    best_acc = 0.0
    for epoch in range(args.epochs):
        model.train()
        start = time.time()
        running_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * x.size(0)
        scheduler.step()

        train_loss = running_loss / len(train_loader.dataset)
        acc = evaluate(model, test_loader, device)
        best_acc = max(best_acc, acc)
        elapsed = time.time() - start
        print(f"epoch {epoch+1}/{args.epochs} loss {train_loss:.4f} acc {acc*100:.2f}% "
              f"best {best_acc*100:.2f}% time {elapsed:.1f}s")

        torch.save(model.state_dict(), os.path.join(args.checkpoint_dir, "last.pt"))
        if acc == best_acc:
            torch.save(model.state_dict(), os.path.join(args.checkpoint_dir, "best.pt"))

    print(f"final best accuracy: {best_acc*100:.2f}%")


if __name__ == "__main__":
    main()
