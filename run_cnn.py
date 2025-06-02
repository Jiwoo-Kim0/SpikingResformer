from IPython import get_ipython
from IPython.display import display

import os
import random
import time
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from timm.data import create_transform
from tqdm import tqdm
from matplotlib import pyplot as plt
from torch.cuda.amp import autocast, GradScaler



# Reproducibility
seed = 12450
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Device setup
#os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
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
train_dataset = datasets.ImageFolder("./dataset/ASL/train", transform=transform_train)
test_dataset = datasets.ImageFolder("./dataset/ASL/val_fixed", transform=transform_val)

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

    train_losses = []
    test_accuracies = []

    for epoch in range(epochs):
        model.train()
        running_loss, correct_top1, correct_top5, total = 0.0, 0, 0, 0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")

        for images, labels in progress_bar:
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

        acc1 = 100 * correct_top1 / total
        acc5 = 100 * correct_top5 / total
        train_losses.append(running_loss / len(train_loader))
        test_accuracies.append(acc1)
        print(f"Train Acc@1: {acc1:.2f}%, Acc@5: {acc5:.2f}%")

        model.eval()
        correct_top1 = correct_top5 = total = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                if isinstance(outputs, dict): outputs = outputs['logits']
                if isinstance(outputs, tuple): outputs = outputs[0]

                _, pred_top1 = outputs.topk(1, dim=1)
                _, pred_top5 = outputs.topk(5, dim=1)
                correct_top1 += (pred_top1.squeeze() == labels).sum().item()
                correct_top5 += sum([labels[i] in pred_top5[i] for i in range(len(labels))])
                total += labels.size(0)

        acc1 = 100 * correct_top1 / total
        acc5 = 100 * correct_top5 / total
        print(f"Test Acc@1: {acc1:.2f}%, Acc@5: {acc5:.2f}%")

    results[model_name] = {
        'loss': train_losses,
        'accuracy': test_accuracies,
        'final_accuracy': test_accuracies[-1]
    }