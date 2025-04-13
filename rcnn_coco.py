#!/usr/bin/env python3
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.ops import MultiScaleRoIAlign
from torchvision.datasets import CocoDetection
from torchvision.transforms import functional as TVF
from torchmetrics.detection.mean_ap import MeanAveragePrecision
import random
import time
from datetime import datetime
import os
from PIL import Image
import numpy as np

# ----------------------- General ------------------------
# Authors: Shabnam Tajik and Mattia Taiana
# Modified for conventional Faster-RCNN
# --------------------------------------------------------

# -------------------- Global Config ---------------------
# Hyperparameters
subset_size = 20000

# Training fundamentals
batch_size = 4
num_epochs = 20
num_classes = 90
base_lr = 0.0005          # For non-backbone parameters
backbone_lr = 0.00005     # 10x smaller than base_lr

# Optimization constraints
max_grad_norm = 1.0       # Global gradient clipping
# --------------------------------------------------------

# ------------------ Logging Utilities -------------------
def log(message):
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}] {message}")
# --------------------------------------------------------

# --------------- Feature Extraction Backbone ------------
class Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        resnet = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V1)
        for module in resnet.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.eval()
                module.weight.requires_grad_(False)
                module.bias.requires_grad_(False)
        self.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        self.out_channels = 2048

    def forward(self, x):
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return {'0': x}
# --------------------------------------------------------

# ------------ COCO Dataset Adapter & Filtering ----------
class FilteredCocoDetection(CocoDetection):
    def __init__(self, root, annFile, transforms):
        super().__init__(root, annFile, transforms=None)
        self._transforms = transforms
        self.ids = sorted(self.coco.getImgIds())[:subset_size]

    def __getitem__(self, index):
        while True:
            try:
                image_id = self.ids[index]
                image = Image.open(os.path.join(self.root, self.coco.loadImgs(image_id)[0]['file_name'])).convert('RGB')
                ann_ids = self.coco.getAnnIds(imgIds=image_id)
                target = self.coco.loadAnns(ann_ids)

                rcnn_target = {
                    'boxes': torch.tensor(
                        [[obj['bbox'][0], obj['bbox'][1],
                          obj['bbox'][0] + obj['bbox'][2],
                          obj['bbox'][1] + obj['bbox'][3]]
                        for obj in target if 'bbox' in obj and len(obj['bbox']) > 0],
                        dtype=torch.float32),
                    'labels': torch.tensor(
                        [obj['category_id'] - 1 for obj in target
                         if 'bbox' in obj and len(obj['bbox']) > 0],
                        dtype=torch.int64)
                }

                if len(rcnn_target['boxes']) == 0:
                    raise ValueError("No valid boxes")

                if self._transforms is not None:
                    image, rcnn_target = self._transforms(image, rcnn_target)

                return image, rcnn_target
            except Exception as e:
                index = random.randint(0, len(self)-1)
# --------------------------------------------------------

# ----------- Data Augmentation Pipeline -----------------
class Compose:
    def __init__(self, transforms):
        self.transforms = transforms
    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target

class Resize:
    def __init__(self, size):
        self.size = size
    def __call__(self, image, target):
        original_width, original_height = image.size
        image = TVF.resize(image, self.size)
        new_width, new_height = image.size
        x_scale = new_width / original_width
        y_scale = new_height / original_height
        target['boxes'][:, [0, 2]] *= x_scale
        target['boxes'][:, [1, 3]] *= y_scale
        return image, target

class RandomHorizontalFlip:
    def __init__(self, prob=0.5):
        self.prob = prob
    def __call__(self, image, target):
        if random.random() < self.prob:
            image = TVF.hflip(image)
            width = image.size[0]
            boxes = target['boxes']
            boxes[:, [0, 2]] = width - boxes[:, [2, 0]]
            target['boxes'] = boxes
        return image, target

class ToTensor:
    def __call__(self, image, target):
        return TVF.to_tensor(image), target

def get_transform(train):
    transforms = [Resize((800, 800))]
    if train:
        transforms.append(RandomHorizontalFlip(0.5))
    transforms.append(ToTensor())
    return Compose(transforms)
# --------------------------------------------------------

# ------------ Model Architecture Factory ----------------
def create_model():
    try:
        log("Initializing model...")
        backbone = Backbone()

        anchor_generator = AnchorGenerator(
            sizes=((32, 64, 128, 256, 512),),
            aspect_ratios=((0.5, 1.0, 2.0),)
        )

        roi_pooler = MultiScaleRoIAlign(
            featmap_names=['0'], output_size=7, sampling_ratio=2
        )

        model = FasterRCNN(
            backbone,
            num_classes=num_classes,
            rpn_anchor_generator=anchor_generator,
            box_roi_pool=roi_pooler,
            box_score_thresh=0.01,
            box_nms_thresh=0.3,
            box_detections_per_img=100
        ).to(device)

        log("Model initialized successfully")
        return model
    except Exception as e:
        log(f"Model initialization failed: {str(e)}")
        raise
# --------------------------------------------------------

# ------------ Training State Persistence ----------------
CHECKPOINT_DIR = "checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

def save_checkpoint(epoch, model, optimizer, losses, is_best=False):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': losses
    }
    filename = f"checkpoint_epoch_{epoch}.pt" if not is_best else "checkpoint_best.pt"
    path = os.path.join(CHECKPOINT_DIR, filename)
    torch.save(checkpoint, path)
    log(f"Checkpoint saved to {path}")
# --------------------------------------------------------

# --------------- Core Training Machinery ----------------
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log(f"Using device: {device}")

    try:
        log("=== Starting Training ===")
        model = create_model()
        optimizer = torch.optim.AdamW([
            {'params': model.backbone.parameters(), 'lr': backbone_lr},
            {'params': [p for n,p in model.named_parameters() if 'backbone' not in n], 'lr': base_lr}
        ], weight_decay=1e-4)

        dataset = FilteredCocoDetection(
            root='coco/train2017',
            annFile='coco/annotations/instances_train2017.json',
            transforms=get_transform(train=True))

        data_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collate_fn
        )

        for epoch in range(num_epochs):
            model.train()
            epoch_start = time.time()
            total_loss = 0.0
            batch_count = 0

            log(f"\nEpoch {epoch+1}/{num_epochs} starting")
            for batch_idx, (images, targets) in enumerate(data_loader):
                try:
                    images = [img.to(device) for img in images]
                    targets = [{
                        'boxes': t['boxes'].to(device),
                        'labels': t['labels'].to(device)
                    } for t in targets]

                    loss_dict = model(images, targets)
                    losses = sum(loss for loss in loss_dict.values())

                    optimizer.zero_grad()
                    losses.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                    optimizer.step()

                    total_loss += losses.item()
                    batch_count += 1

                    if batch_idx % 10 == 0:
                        log(f"Batch {batch_idx:04d}/{len(data_loader):04d} | "
                            f"Loss: {losses.item():.4f} | "
                            f"Class: {loss_dict['loss_classifier'].item():.4f} | "
                            f"Box: {loss_dict['loss_box_reg'].item():.4f}")

                except Exception as e:
                    log(f"Error in batch {batch_idx}: {str(e)}")
                    continue

            avg_loss = total_loss / batch_count
            epoch_time = (time.time() - epoch_start) / 60
            log(f"Epoch {epoch+1} Complete | Avg Loss: {avg_loss:.4f} | Time: {epoch_time:.1f}min")

            evaluate_model(model)
            save_checkpoint(epoch, model, optimizer, avg_loss)

    except Exception as e:
        log(f"Training failed: {str(e)}")
        raise
# --------------------------------------------------------

# ---------- Batch Collation & Type Handling -------------
def collate_fn(batch):
    images = []
    targets = []
    for img, target in batch:
        if not isinstance(img, torch.Tensor):
            img = TVF.to_tensor(img)
        images.append(img)
        targets.append(target)
    return images, targets
# --------------------------------------------------------

# ---------- Model Validation & Metrics ------------------
def evaluate_model(model):
    log("\n=== Starting Evaluation ===")
    metric = MeanAveragePrecision()
    val_dataset = FilteredCocoDetection(
        root='coco/val2017',
        annFile='coco/annotations/instances_val2017.json',
        transforms=get_transform(train=False))
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    model.eval()
    with torch.no_grad():
        for images, targets in val_loader:
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            try:
                predictions = model(images)
                metric.update(predictions, targets)
            except Exception as e:
                log(f"Evaluation error: {str(e)}")
                continue

    result = metric.compute()
    log(f"Evaluation mAP: {result['map']:.4f} | mAP50: {result['map_50']:.4f}")
# --------------------------------------------------------

# ------------------ Execution Entrypoint ----------------
if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    main()
# --------------------------------------------------------
