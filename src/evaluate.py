import argparse

import torch
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader

from model import wrn_28_10, wrn_40_2
from train import CIFAR10_MEAN, CIFAR10_STD, CIFAR100_MEAN, CIFAR100_STD, evaluate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["cifar10", "cifar100"], default="cifar10")
    parser.add_argument("--model", choices=["wrn_28_10", "wrn_40_2"], default="wrn_40_2")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-dir", default="./data")
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mean, std = (CIFAR100_MEAN, CIFAR100_STD) if args.dataset == "cifar100" else (CIFAR10_MEAN, CIFAR10_STD)
    num_classes = 100 if args.dataset == "cifar100" else 10

    tf = T.Compose([T.ToTensor(), T.Normalize(mean, std)])
    cls = torchvision.datasets.CIFAR100 if args.dataset == "cifar100" else torchvision.datasets.CIFAR10
    test_set = cls(root=args.data_dir, train=False, download=True, transform=tf)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=4)

    model = (wrn_28_10 if args.model == "wrn_28_10" else wrn_40_2)(num_classes=num_classes).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))

    acc = evaluate(model, test_loader, device)
    print(f"test accuracy: {acc*100:.2f}%")


if __name__ == "__main__":
    main()
