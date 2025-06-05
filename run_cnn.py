from datetime import datetime
import logging
import os
import random
import time
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from timm.data import create_transform
from timm.scheduler import create_scheduler_v2
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler
from types import SimpleNamespace
import matplotlib.pyplot as plt

# Set up logger
logger = logging.getLogger("train_logger")
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter("[%(asctime)s][%(levelname)s] %(message)s", "%Y-%m-%d %H:%M:%S")
handler = logging.StreamHandler()
handler.setFormatter(formatter)
logger.handlers = [handler]

# Reproducibility
seed = 12450
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Device
device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")

# Hyperparameters
epochs = 20
batch_size = 16
learning_rate = 5e-4
weight_decay = 0.01
label_smoothing = 0.1
optimizer_type = "adamw"
input_size = (3, 32, 32)
num_classes = 27

# Transforms
transform_train = create_transform(
    input_size=input_size,
    is_training=True,
    auto_augment='rand-m7-n1-mstd0.5-inc1',
    interpolation='bicubic',
    mean=(0.4914, 0.4822, 0.4465),
    std=(0.2023, 0.1994, 0.2010),
)

transform_val = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010))
])

# Dataset
train_dataset = datasets.ImageFolder("/dev/hdd/bcl_guest/ASL/train", transform=transform_train)
test_dataset = datasets.ImageFolder("/dev/hdd/bcl_guest/ASL/val_fixed", transform=transform_val)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=16, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=16, pin_memory=True)

# Models
model_dict = {
    'ResNet18': lambda: models.resnet18(weights=models.ResNet18_Weights.DEFAULT),
    'GoogLeNet': lambda: models.googlenet(weights=models.GoogLeNet_Weights.DEFAULT, aux_logits=True),
    'MobileNetV2': lambda: models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT),
}

scaler = GradScaler()
results = {}

def train_and_evaluate(model_name):
    print(f"Training {model_name}")
    model = model_dict[model_name]()

    if hasattr(model, 'fc'):
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif hasattr(model, 'classifier') and isinstance(model.classifier, nn.Sequential):
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)

    model.to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    args = SimpleNamespace(
        epochs=epochs,
        cooldown_epochs=10,
        min_lr=1e-5,
        warmup_lr=1e-5,
        warmup_epochs=3
    )

    lr_scheduler, _ = create_scheduler_v2(
        optimizer,
        sched='cosine',
        num_epochs=args.epochs,
        cooldown_epochs=args.cooldown_epochs,
        min_lr=args.min_lr,
        warmup_lr=args.warmup_lr,
        warmup_epochs=args.warmup_epochs,
    )

    logger.info(f"[Train]")
    for epoch in range(epochs):
        model.train()
        logger.info(f"Epoch [{epoch}] Start, lr {optimizer.param_groups[0]['lr']:.6f}")
        start_time = time.time()
        running_loss, correct_top1, correct_top5, total = 0.0, 0, 0, 0

        for idx, (images, labels) in enumerate(train_loader, 1):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            with autocast():
                outputs = model(images)
                if isinstance(outputs, dict): outputs = outputs['logits']
                if isinstance(outputs, tuple): outputs = outputs[0]
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()
            _, pred_top1 = outputs.topk(1, dim=1)
            _, pred_top5 = outputs.topk(5, dim=1)
            correct_top1 += (pred_top1.squeeze() == labels).sum().item()
            correct_top5 += sum([labels[i] in pred_top5[i] for i in range(len(labels))])
            total += labels.size(0)

            if idx % max(1, len(train_loader) // 5) == 0:
                it_per_s = idx / (time.time() - start_time)
                acc1 = 100 * correct_top1 / total
                acc5 = 100 * correct_top5 / total
                logger.debug(f"[{idx}/{len(train_loader)}] it/s: {it_per_s:.5f}, loss: {loss.item():.5f}, acc@1: {acc1:.5f}, acc@5: {acc5:.5f}")

        elapsed = time.time() - start_time
        logger.debug(f"Train spent: {time.strftime('%H:%M:%S', time.gmtime(elapsed))}")

        model.eval()
        total, correct_top1, correct_top5 = 0, 0, 0
        test_loss = 0.0
        logger.debug("Test start")
        start_test = time.time()
        with torch.no_grad():
            for idx, (images, labels) in enumerate(test_loader, 1):
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                if isinstance(outputs, dict): outputs = outputs['logits']
                if isinstance(outputs, tuple): outputs = outputs[0]

                loss = criterion(outputs, labels)
                test_loss += loss.item()

                _, pred_top1 = outputs.topk(1, dim=1)
                _, pred_top5 = outputs.topk(5, dim=1)
                correct_top1 += (pred_top1.squeeze() == labels).sum().item()
                correct_top5 += sum([labels[i] in pred_top5[i] for i in range(len(labels))])
                total += labels.size(0)

        elapsed_test = time.time() - start_test
        acc1 = 100 * correct_top1 / total
        acc5 = 100 * correct_top5 / total
        logger.info(f"Test loss: {test_loss/len(test_loader):.5f}, Acc@1: {acc1:.5f}, Acc@5: {acc5:.5f}")

        if lr_scheduler is not None:
            lr_scheduler.step(epoch + 1)

    logger.info("Training completed.")

def main():
    for model_name in model_dict:
        start = time.time()
        train_and_evaluate(model_name)
        print(f"{model_name} done in {time.time() - start:.1f}s\n")

if __name__ == "__main__":
    main()