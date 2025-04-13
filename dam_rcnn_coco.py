#!/usr/bin/env python3
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
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
# (this code was written taking into account inputs from generative models)
#
# ----------------- Project Description ------------------
# Final Project of CSC478 course at the University of Toronto focussing on Dense and Hierarchical Associative Memory
# for Object Detection in R-CNN and enhancing robustness in complex object detection scenarios.
# The goal is to analyze the performance of Dense Associative Memory in a hybrid classification setup:
# 1. Backbone CNN feature extraction (unchanged to original Faster-RCNN)
# 2. Region Proposal Network and RoI Pooling (unchanged to original Faster-RCNN)
#    -> DAM not suitable for localization
# 3. DAM Head replacing classifier (learns class-prototypes and improves rare-class performance)
# 4. Box Head predicts box offsets (unchanged)
# --------------------------------------------------------

# -------------------- Global Config ---------------------
# Hyperparameters and experiment settings balancing model performance,
# training stability, and rare class emphasis (cell phone/hair drier)

# Dataset Size
subset_size = 20000

# Training fundamentals
batch_size = 4
num_epochs = 20
num_classes = 90
base_lr = 0.0005          # For non-backbone parameters
backbone_lr = 0.00005     # 10x smaller than base_lr

# DAM-specific parameters
initial_prototype_scale = 0.001
max_prototype_norm = 3.0
min_prototype_norm = 0.3
proto_reg_weight = 0.1     # Weight for prototype L2 regularization

# Optimization constraints
max_grad_norm = 1.0       # Global gradient clipping
dam_grad_scale = 0.1      # Additional scaling for prototype gradients
temp_grad_scale = 0.1     # Temperature parameter learning rate scaling
power_grad_clip = 0.001   # Max gradient magnitude for power parameter

# Stabilization parameters
similarity_clip = 10.0    # Absolute value clip for similarity scores
min_temperature = 0.1     # prevents division explosions
max_temperature = 5.0
power_range = (0.5, 2.0)
# --------------------------------------------------------

# ------------------ Logging Utilities -------------------
# Centralized logging system for tracking training progress and debugging,
# critical for long-running experiments and reproducibility
def log(message):
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}] {message}")
# --------------------------------------------------------

# --------------- Feature Extraction Backbone ------------
# Frozen ResNet50 architecture providing spatial feature maps while
# preserving pretrained ImageNet knowledge through BatchNorm freezing
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

# ---------- Dense Associative Memory Classifier ---------
# Replacement for traditional classifier head using learnable prototypes
# and stabilized similarity metrics to enhance rare-class discrimination
class DAMFastRCNNPredictor(nn.Module):
    def __init__(self, in_features, num_classes=num_classes):
        super().__init__()
        self.prototypes = nn.Parameter(torch.randn(num_classes, in_features) * initial_prototype_scale)
        self.log_temperature = nn.Parameter(torch.tensor(-1.0))
        self.log_power = nn.Parameter(torch.tensor(-0.5))

        self.bbox_pred = nn.Linear(in_features, num_classes * 4)

        nn.init.normal_(self.bbox_pred.weight, std=0.001)
        nn.init.constant_(self.bbox_pred.bias, 0)

    def forward(self, x):
        with torch.no_grad():
            prototype_norms = torch.norm(self.prototypes, p=2, dim=1, keepdim=True)
            scale_factors = torch.clamp(prototype_norms, 0.3, 3.0) / (prototype_norms + 1e-8)
            self.prototypes.mul_(scale_factors)

        prototypes_norm = F.normalize(self.prototypes, p=2, dim=1, eps=1e-8)
        x_norm = F.normalize(x, p=2, dim=1, eps=1e-8)

        temperature = torch.exp(self.log_temperature.clamp(min=np.log(min_temperature), max=np.log(max_temperature)))
        power = torch.exp(self.log_power.clamp(min=np.log(power_range[0]), max=np.log(power_range[1])))

        sim = torch.mm(x_norm, prototypes_norm.T) / (temperature + 1e-8)
        sim = torch.clamp(sim, -similarity_clip, similarity_clip)

        abs_sim = sim.abs() + 1e-8
        powered_sim = torch.exp(power * torch.log(abs_sim))
        cls_logits = torch.sign(sim) * powered_sim

        cls_logits = torch.nan_to_num(cls_logits, nan=0.0, posinf=20.0, neginf=-20.0)
        cls_logits[:, 80:] = -1e8 # avoid -inf

        return cls_logits, self.bbox_pred(x)
# --------------------------------------------------------

# ------------ COCO Dataset Adapter & Filtering ----------
# Custom data loader implementing COCO-specific format conversion,
# invalid sample filtering, and size-based dataset subset selection
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
# Image preprocessing and augmentation stack balancing:
# - Resolution normalization (800x800)
# - Horizontal flipping for geometric invariance
# - Tensor conversion for GPU acceleration
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
# Faster R-CNN constructor with custom components:
# 1. Frozen ResNet backbone 2. DAM classifier 3. Modified detection params
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

        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = DAMFastRCNNPredictor(in_features).to(device)

        log("Model initialized successfully")
        for name, param in model.named_parameters():
            if not param.is_cuda:
                print(f"Warning: {name} not on GPU!")

        return model
    except Exception as e:
        log(f"Model initialization failed: {str(e)}")
        raise
# --------------------------------------------------------

# ------------ Training State Persistence ----------------
# Checkpoint system enabling experiment resumption and model selection
# through periodic saves of model parameters and optimizer state
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
def load_checkpoint(path, model, optimizer):
    if os.path.exists(path):
        checkpoint = torch.load(path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        log(f"Resuming from epoch {checkpoint['epoch']} with loss {checkpoint['loss']}")
        return checkpoint['epoch']
    return 0
# --------------------------------------------------------

# --------------- Core Training Machinery ----------------
# Main optimization loop implementing:
# - Batch-wise forward/backward passes
# - DAM-specific gradient control
# - Training stability mechanisms
# - Progress monitoring and metrics logging
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    g = torch.Generator(device='cpu')
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
            collate_fn=collate_fn,
            generator=g
        )

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            log(f"Initial GPU Memory: Allocated={torch.cuda.memory_allocated()/1e6:.2f}MB, Reserved={torch.cuda.memory_reserved()/1e6:.2f}MB")

        for epoch in range(num_epochs):
            model.train()
            epoch_start = time.time()
            total_loss = 0.0
            batch_count = 0

            log(f"\nEpoch {epoch+1}/10 starting")
            log(f"Dataset size: {len(dataset)}")
            log(f"DataLoader length: {len(data_loader)}")

            for batch_idx, (images, targets) in enumerate(data_loader):
                try:
                    # Move data to device
                    images = [img.to(device) if torch.is_tensor(img) else TVF.to_tensor(img).to(device) for img in images]
                    targets = [{
                        'boxes': t['boxes'].to(device),
                        'labels': t['labels'].to(device)
                    } for t in targets]

                    # Forward pass
                    loss_dict = model(images, targets)
                    prototypes = model.roi_heads.box_predictor.prototypes
                    proto_reg = 0.1 * torch.norm(prototypes, p=2)
                    losses = sum(loss for loss in loss_dict.values()) + proto_reg

                    # Debug NaN/inf values
                    if torch.isnan(losses) or torch.isinf(losses):
                        log(f"Invalid loss detected at batch {batch_idx}")
                        log(f"Class loss: {loss_dict['loss_classifier'].item()}")
                        log(f"Box loss: {loss_dict['loss_box_reg'].item()}")
                        with torch.no_grad():
                            # Reset prototypes and scaling parameters
                            model.roi_heads.box_predictor.prototypes.data = (
                                torch.randn_like(model.roi_heads.box_predictor.prototypes) * 0.001
                            )
                            model.roi_heads.box_predictor.log_temperature.data.fill_(-1.0)
                            model.roi_heads.box_predictor.log_power.data.fill_(-0.5)
                        continue

                    # Backward pass
                    optimizer.zero_grad()
                    losses.backward()

                    # DAM-specific gradient control
                    for name, param in model.named_parameters():
                        if 'prototypes' in name:
                            param.grad *= dam_grad_scale  # From config
                            param.grad.clamp_(-0.01, 0.01)
                        elif 'log_temperature' in name:
                            param.grad = param.grad * temp_grad_scale  # From config
                        elif 'log_power' in name:
                            param.grad.clamp_(-power_grad_clip, power_grad_clip)  # From config

                    # Gradient clipping and scaling
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                    for p in model.roi_heads.box_predictor.prototypes:
                        if p.grad is not None:
                            p.grad *= 0.1  # Scale down prototype gradients
                    optimizer.step()
                    for param in model.parameters():
                        if param.grad is not None:
                            noise = torch.randn_like(param.grad) * (0.001 * torch.std(param.grad))
                            param.grad += noise

                    with torch.no_grad():
                        prototypes = model.roi_heads.box_predictor.prototypes
                        norms = torch.norm(prototypes, p=2, dim=1)
                        median_norm = torch.median(norms).item()

                        if not 0.5 < median_norm < 2.0:
                            prototypes.div_(median_norm + 1e-8)

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

            # Epoch summary
            avg_loss = total_loss / batch_count
            epoch_time = (time.time() - epoch_start) / 60
            log(f"Epoch {epoch+1} Complete | "
                f"Avg Loss: {avg_loss:.4f} | "
                f"Time: {epoch_time:.1f}min")

            # Evaluation and checkpointing
            evaluate_model(model)
            save_checkpoint(epoch, model, optimizer, avg_loss)

    except Exception as e:
        log(f"Training failed: {str(e)}")
        raise
# --------------------------------------------------------

# ---------- Batch Collation & Type Handling -------------
# Data loader helper ensuring consistent tensor conversion
# for variable-sized images and annotation dictionaries
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
# Evaluation protocol using COCO-standard mean Average Precision (mAP)
# to quantify detection quality across object categories
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

                # Ensure predictions contain valid boxes
                for pred in predictions:
                    if len(pred['boxes']) == 0:
                        # Create dummy prediction if empty
                        pred['boxes'] = torch.tensor([[0,0,1,1]], device=device)
                        pred['scores'] = torch.tensor([0.01], device=device)
                        pred['labels'] = torch.tensor([0], device=device)

                metric.update(predictions, targets)
            except Exception as e:
                log(f"Evaluation error: {str(e)}")
                continue

    try:
        result = metric.compute()
        log(f"Evaluation mAP: {result['map']:.4f} | mAP50: {result['map_50']:.4f}")
    except Exception as e:
        log(f"Metric computation failed: {str(e)}")
        log("Model is producing invalid predictions")
# ---------------------------------------------------------

# ------------------ Execution Entrypoint -----------------
# Device-aware initialization and training kickoff
if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log(f"Using device: {device}")
    if torch.cuda.is_available():
        log(f"Current GPU: {torch.cuda.get_device_name(0)}")

    main()
# ---------------------------------------------------------
