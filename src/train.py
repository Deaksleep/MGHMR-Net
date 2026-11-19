"""Train and evaluate a DINOv2 multitask model for neuron images.

The model jointly predicts a culture-day group and elastic modulus. Reported
metrics are computed from out-of-fold predictions produced by stratified
cross-validation.
"""

import argparse
import glob
import json
import os
import platform
import random
import copy
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
import tifffile
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LinearRegression


def seed_everything(seed=42):
    """Seed Python, NumPy, and PyTorch random-number generators."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class AddGaussianNoise:
    """Add Gaussian noise to an image tensor and clip it to [0, 1]."""
    def __init__(self, mean=0.0, std=0.012):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        noise = torch.randn(tensor.size()) * self.std + self.mean
        return torch.clamp(tensor + noise, 0.0, 1.0)


def clone_dataset_view(dataset, is_train):
    view = copy.copy(dataset)
    view.is_train = is_train
    return view


def update_ema_state(ema_state, model, decay=0.97):
    current_state = model.state_dict()
    if ema_state is None:
        return {name: value.detach().cpu().clone() for name, value in current_state.items()}
    for name, value in current_state.items():
        value_cpu = value.detach().cpu()
        if torch.is_floating_point(value_cpu):
            ema_state[name].mul_(decay).add_(value_cpu, alpha=1.0 - decay)
        else:
            ema_state[name] = value_cpu.clone()
    return ema_state


def make_label_bins(targets, n_bins=6):
    quantiles = np.linspace(0.0, 1.0, n_bins + 1)
    edges = np.unique(np.quantile(targets, quantiles))
    if len(edges) <= 2:
        return np.zeros(len(targets), dtype=np.int64)
    return np.digitize(targets, edges[1:-1], right=True)


def make_stratification_labels(day_labels, targets, n_bins=4):
    bins = make_label_bins(targets, n_bins=n_bins)
    combined = day_labels * 10 + bins
    _, counts = np.unique(combined, return_counts=True)
    if counts.min() < 5:
        return day_labels
    return combined


class HybridDayModulusDataset(Dataset):
    """TIFF images paired with modulus and culture-day labels."""
    DAY_GROUPS = [
        ("Day_1", (1,)),
        ("Day_4_7", (4, 7)),
        ("Day_10_13", (10, 13)),
    ]
    DAY_CLASSES = [name for name, _ in DAY_GROUPS]
    DAY_TO_INDEX = {
        day: idx
        for idx, (_, days) in enumerate(DAY_GROUPS)
        for day in days
    }

    def __init__(self, img_dir, label_file, target_canvas_size=224, is_train=True, r_threshold=0.75):
        self.img_dir = img_dir
        self.target_canvas_size = target_canvas_size
        self.is_train = is_train
        self.r_threshold = r_threshold

        df = pd.read_excel(label_file)
        required_columns = {"filename", "modulus", "R", "day"}
        missing_columns = required_columns.difference(df.columns)
        if missing_columns:
            raise ValueError(
                f"Label file is missing required columns: {sorted(missing_columns)}"
            )
        img_files = glob.glob(os.path.join(img_dir, "*.tif")) + glob.glob(os.path.join(img_dir, "*.tiff"))
        img_path_dict = {os.path.basename(path): path for path in img_files}

        self.valid_data = []
        for _, row in df.iterrows():
            fname = str(row["filename"]).strip()
            modulus = row["modulus"]
            r_value = row["R"]
            day = row["day"]
            if (
                pd.isna(modulus)
                or modulus <= 0
                or pd.isna(r_value)
                or r_value <= self.r_threshold
                or pd.isna(day)
                or int(day) not in self.DAY_TO_INDEX
                or fname not in img_path_dict
            ):
                continue
            day = int(day)
            self.valid_data.append(
                {
                    "img_path": img_path_dict[fname],
                    "filename": fname,
                    "true_modulus": float(modulus),
                    "day": day,
                    "day_label": self.DAY_TO_INDEX[day],
                    "r_value": float(r_value),
                }
            )

        self.train_transforms = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomRotation(degrees=180),
                transforms.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.95, 1.05), shear=5),
                AddGaussianNoise(mean=0.0, std=0.012),
            ]
        )
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )

    def __len__(self):
        return len(self.valid_data)

    def get_cell_bbox(self, img, margin=15):
        """Estimate a cell-centred box from intensity and edge masks."""
        median_val = np.median(img)
        mad = np.median(np.abs(img - median_val))
        intensity_mask = img > (median_val + 0.75 * mad)
        gy, gx = np.gradient(img.astype(np.float32))
        edge = np.sqrt(gx ** 2 + gy ** 2)
        edge_mask = edge > np.percentile(edge, 90)
        mask = intensity_mask | edge_mask

        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)
        if not np.any(rows) or not np.any(cols):
            return 0, img.shape[0], 0, img.shape[1]

        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]
        rmin, rmax = max(0, rmin - margin), min(img.shape[0], rmax + margin)
        cmin, cmax = max(0, cmin - margin), min(img.shape[1], cmax + margin)
        return rmin, rmax, cmin, cmax

    def pad_to_canvas(self, img_tensor):
        """Centre-crop or zero-pad a CHW tensor to a square canvas."""
        _, h, w = img_tensor.shape
        target = self.target_canvas_size
        if h > target or w > target:
            top = max(0, (h - target) // 2)
            left = max(0, (w - target) // 2)
            img_tensor = img_tensor[:, top:top + target, left:left + target]
            _, h, w = img_tensor.shape

        pad_top = (target - h) // 2
        pad_bottom = target - h - pad_top
        pad_left = (target - w) // 2
        pad_right = target - w - pad_left
        if pad_top > 0 or pad_bottom > 0 or pad_left > 0 or pad_right > 0:
            img_tensor = F.pad(
                img_tensor.unsqueeze(0),
                (pad_left, pad_right, pad_top, pad_bottom),
                mode="constant",
                value=0.0,
            ).squeeze(0)
        return img_tensor

    def __getitem__(self, idx):
        item = self.valid_data[idx]
        raw_img = tifffile.imread(item["img_path"]).astype(np.float32)
        if raw_img.ndim == 3 and raw_img.shape[-1] == 1:
            raw_img = raw_img[:, :, 0]
        if raw_img.ndim == 3:
            raw_img = raw_img.mean(axis=-1)

        img = (raw_img - np.mean(raw_img)) / (np.std(raw_img) + 1e-8)
        rmin, rmax, cmin, cmax = self.get_cell_bbox(img, margin=15)
        img = img[rmin:rmax, cmin:cmax]

        p1, p99 = np.percentile(img, 1), np.percentile(img, 99)
        img_normed = np.clip((img - p1) / (p99 - p1 + 1e-8), 0.0, 1.0)
        img_tensor = torch.from_numpy(np.stack([img_normed, img_normed, img_normed], axis=0)).float()
        img_tensor = self.pad_to_canvas(img_tensor)

        if self.is_train:
            img_tensor = self.train_transforms(img_tensor)
        img_tensor = self.normalize(img_tensor)

        patch_mask = torch.ones(
            (1, self.target_canvas_size // 14, self.target_canvas_size // 14),
            dtype=torch.float32,
        )
        return (
            img_tensor,
            patch_mask,
            torch.tensor(item["day_label"], dtype=torch.long),
            torch.tensor(item["true_modulus"], dtype=torch.float32),
            item["filename"],
            item["day"],
        )


class DINOv2HybridDayModulus(nn.Module):
    """DINOv2 encoder with classification and regression heads."""
    def __init__(
        self,
        num_day_classes=3,
        finetune_last_blocks=1,
        backbone_repo="facebookresearch/dinov2",
    ):
        super().__init__()
        self.backbone = torch.hub.load(backbone_repo, "dinov2_vits14")
        embed_dim = 384

        for param in self.backbone.parameters():
            param.requires_grad = False
        self.backbone_trainable = False
        if finetune_last_blocks > 0 and hasattr(self.backbone, "blocks"):
            for block in self.backbone.blocks[-finetune_last_blocks:]:
                for param in block.parameters():
                    param.requires_grad = True
                    self.backbone_trainable = True
            if hasattr(self.backbone, "norm"):
                for param in self.backbone.norm.parameters():
                    param.requires_grad = True
                    self.backbone_trainable = True
        self.backbone.eval()

        self.feature_pooler = nn.Sequential(
            nn.Linear(embed_dim * 3, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.15),
        )
        self.day_head = nn.Sequential(
            nn.Linear(256, 96),
            nn.LayerNorm(96),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(96, num_day_classes),
        )
        self.regressor = nn.Sequential(
            nn.Linear(256 + num_day_classes + 1, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(0.20),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )
        self.register_buffer("stage_values", torch.linspace(0.0, 1.0, steps=num_day_classes).view(1, -1))
        self.register_buffer("day_mean_z", torch.zeros(1, num_day_classes))

    def set_day_mean_z(self, day_mean_z):
        values = torch.as_tensor(day_mean_z, dtype=torch.float32).view(1, -1)
        self.day_mean_z = values.to(self.day_mean_z.device)

    def forward(self, x_img, patch_mask=None):
        with torch.set_grad_enabled(self.backbone_trainable and self.training):
            features = self.backbone.forward_features(x_img)
            cls_token = features["x_norm_clstoken"]
            patch_tokens = features["x_norm_patchtokens"]

            if patch_mask is not None:
                patch_mask = patch_mask.flatten(1).to(patch_tokens.device).bool()
                empty_mask = patch_mask.sum(dim=1) == 0
                if empty_mask.any():
                    patch_mask[empty_mask] = True
                valid = patch_mask.unsqueeze(-1)
                patch_avg = patch_tokens.masked_fill(~valid, 0.0).sum(dim=1)
                patch_avg = patch_avg / patch_mask.sum(dim=1, keepdim=True).clamp_min(1)
                patch_max = patch_tokens.masked_fill(~valid, -1e4).max(dim=1)[0]
            else:
                patch_avg = patch_tokens.mean(dim=1)
                patch_max = patch_tokens.max(dim=1)[0]

        img_vec = self.feature_pooler(torch.cat([cls_token, patch_avg, patch_max], dim=1))
        day_logits = self.day_head(img_vec)
        day_probs = torch.softmax(day_logits, dim=1)
        expected_stage = (day_probs * self.stage_values.to(day_probs.device)).sum(dim=1, keepdim=True)
        day_context = day_probs.detach()
        day_prior_z = (day_context * self.day_mean_z.to(day_context.device)).sum(dim=1)
        reg_input = torch.cat([img_vec, day_context, expected_stage.detach()], dim=1)
        residual_z = self.regressor(reg_input).squeeze(1)
        modulus_z = day_prior_z + residual_z
        return day_logits, modulus_z, residual_z


def weighted_regression_loss(pred_z, target_z, weights=None, beta=0.5, mse_weight=0.35):
    """Blend Smooth L1 and MSE, optionally using per-sample weights."""
    pred_z = pred_z.view(-1)
    target_z = target_z.view(-1)
    smooth = F.smooth_l1_loss(pred_z, target_z, beta=beta, reduction="none")
    mse = (pred_z - target_z) ** 2
    loss = (1.0 - mse_weight) * smooth + mse_weight * mse
    if weights is not None:
        loss = loss * weights.view(-1)
    return loss.mean()


def parse_args():
    """Define all experiment settings in a discoverable CLI."""
    parser = argparse.ArgumentParser(
        description="Train the DINOv2 neuron day/modulus multitask model."
    )
    parser.add_argument("--image-dir", type=Path, default=Path("data/images"))
    parser.add_argument("--label-file", type=Path, default=Path("data/labels.xlsx"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--r-threshold", type=float, default=0.75)
    parser.add_argument("--canvas-size", type=int, default=224)
    parser.add_argument("--cls-loss-weight", type=float, default=0.20)
    parser.add_argument("--residual-l2-weight", type=float, default=0.02)
    parser.add_argument("--finetune-last-blocks", type=int, default=1)
    parser.add_argument(
        "--backbone-repo",
        default="facebookresearch/dinov2",
        help=(
            "PyTorch Hub repository. For archival reproduction, use "
            "facebookresearch/dinov2:<commit>."
        ),
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=("auto", "cpu", "cuda", "mps"),
        help="Training device. Auto prefers CUDA, then Apple MPS, then CPU.",
    )
    return parser.parse_args()


def resolve_device(requested):
    """Resolve an explicit or automatically selected PyTorch device."""
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main(args=None):
    args = parse_args() if args is None else args
    seed_everything(args.seed)
    img_dir = str(args.image_dir)
    label_file = str(args.label_file)
    output_dir = str(args.output_dir)
    r_threshold = args.r_threshold
    epochs = args.epochs
    batch_size = args.batch_size
    cls_loss_weight = args.cls_loss_weight
    residual_l2_weight = args.residual_l2_weight
    device = resolve_device(args.device)

    if not args.image_dir.is_dir():
        raise FileNotFoundError(
            f"Image directory not found: {args.image_dir}. "
            "Run `python scripts/prepare_data.py` first."
        )
    if not args.label_file.is_file():
        raise FileNotFoundError(f"Label file not found: {args.label_file}")
    if args.canvas_size % 14 != 0:
        raise ValueError("--canvas-size must be divisible by the DINOv2 patch size (14).")
    if args.folds < 2:
        raise ValueError("--folds must be at least 2.")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    run_config = vars(args).copy()
    run_config.update(
        {
            "image_dir": img_dir,
            "label_file": label_file,
            "output_dir": output_dir,
            "software_versions": {
                "python": platform.python_version(),
                "numpy": np.__version__,
                "pandas": pd.__version__,
                "pytorch": torch.__version__,
                "tifffile": tifffile.__version__,
            },
        }
    )
    with open(args.output_dir / "run_config.json", "w", encoding="utf-8") as file:
        json.dump(run_config, file, indent=2)
    print(f">>> Device: {device}")

    dataset = HybridDayModulusDataset(
        img_dir,
        label_file,
        target_canvas_size=args.canvas_size,
        is_train=False,
        r_threshold=r_threshold,
    )
    if len(dataset) == 0:
        raise ValueError("No image/label pairs passed the data filters.")
    targets = np.array([item["true_modulus"] for item in dataset.valid_data], dtype=np.float32)
    day_labels = np.array([item["day_label"] for item in dataset.valid_data], dtype=np.int64)
    strat_labels = make_stratification_labels(day_labels, targets, n_bins=4)
    class_names = HybridDayModulusDataset.DAY_CLASSES
    class_counts = np.bincount(day_labels, minlength=len(class_names))

    print(f">>> R threshold: {r_threshold}; total images used: {len(dataset)}")
    print(">>> Day class order:", class_names)
    print(">>> Day class counts:", dict(zip(class_names, class_counts.tolist())))
    print(">>> Modulus by day group:")
    for idx, name in enumerate(class_names):
        vals = targets[day_labels == idx]
        print(f"    {name}: n={len(vals)}, mean={np.mean(vals):.2f}, std={np.std(vals):.2f}, range={np.min(vals):.2f}-{np.max(vals):.2f}")

    if class_counts.min() < args.folds:
        raise ValueError(
            f"Each day class needs at least {args.folds} samples; "
            f"counts={class_counts.tolist()}."
        )
    skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    oof_reg_preds, oof_targets = [], []
    oof_calibrated_preds, oof_day_aware_calibrated_preds = [], []
    oof_day_true, oof_day_pred, oof_day_prob = [], [], []
    oof_fnames, oof_days = [], []
    oof_folds = []
    fold_records = []
    training_history = []
    history_csv_path = os.path.join(output_dir, "hybrid_multitask_training_history.csv")

    for fold, (train_idx, val_idx) in enumerate(skf.split(np.arange(len(dataset)), strat_labels)):
        print(f"\n---> Training Fold {fold + 1}/{args.folds} <---")
        train_base = clone_dataset_view(dataset, is_train=True)
        val_base = clone_dataset_view(dataset, is_train=False)
        train_loader = DataLoader(
            Subset(train_base, train_idx),
            batch_size=batch_size,
            shuffle=True,
            drop_last=len(train_idx) >= batch_size,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
        )
        val_loader = DataLoader(
            Subset(val_base, val_idx),
            batch_size=batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
        )

        train_targets = targets[train_idx]
        train_log_targets = np.log(np.maximum(train_targets, 1e-3))
        target_mean = float(np.mean(train_log_targets))
        target_std = float(np.std(train_log_targets) + 1e-8)
        mean_t = torch.tensor(target_mean, dtype=torch.float32, device=device)
        std_t = torch.tensor(target_std, dtype=torch.float32, device=device)

        train_day_labels = day_labels[train_idx]
        train_counts = np.bincount(train_day_labels, minlength=len(class_names))
        class_weights = len(train_day_labels) / (len(class_names) * np.maximum(train_counts, 1))
        class_weights_t = torch.tensor(class_weights, dtype=torch.float32, device=device)
        day_mean_z = []
        for class_idx in range(len(class_names)):
            class_targets = train_targets[train_day_labels == class_idx]
            if len(class_targets) == 0:
                class_mean = target_mean
            else:
                class_mean = float(np.mean(np.log(np.maximum(class_targets, 1e-3))))
            day_mean_z.append((class_mean - target_mean) / target_std)

        hist_bins = np.unique(np.quantile(train_targets, np.linspace(0.0, 1.0, 7)))
        if len(hist_bins) < 3:
            hist_bins = 6
        hist_counts, hist_bins = np.histogram(train_targets, bins=hist_bins)
        hist_counts = np.maximum(hist_counts, 1)
        inv_weights = 1.0 / np.sqrt(hist_counts)
        inv_weights = inv_weights / np.mean(inv_weights)

        model = DINOv2HybridDayModulus(
            num_day_classes=len(class_names),
            finetune_last_blocks=args.finetune_last_blocks,
            backbone_repo=args.backbone_repo,
        ).to(device)
        model.set_day_mean_z(day_mean_z)
        head_params = list(model.feature_pooler.parameters()) + list(model.day_head.parameters()) + list(model.regressor.parameters())
        backbone_params = [p for p in model.backbone.parameters() if p.requires_grad]
        optimizer = optim.AdamW(
            [
                {"params": head_params, "lr": 3e-4, "weight_decay": 8e-3},
                {"params": backbone_params, "lr": 1e-5, "weight_decay": 1e-4},
            ]
        )
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
        ema_state = None
        ema_start_epoch = 30
        final_model_path = os.path.join(output_dir, f"final_model_dinov2_hybrid_fold_{fold}.pth")

        for epoch in range(epochs):
            model.train()
            train_loss_sum = 0.0
            train_reg_sum = 0.0
            train_cls_sum = 0.0
            train_correct = 0
            train_total = 0

            for imgs, patch_masks, y_day, modulus, _, _ in train_loader:
                imgs = imgs.to(device)
                patch_masks = patch_masks.to(device)
                y_day = y_day.to(device)
                y_value = modulus.to(device).float()
                y_true_z = ((torch.log(y_value.clamp_min(1e-3)) - mean_t) / std_t).view(-1)

                batch_weights = []
                for val in modulus.numpy():
                    bin_idx = np.digitize(val, hist_bins) - 1
                    bin_idx = np.clip(bin_idx, 0, len(hist_counts) - 1)
                    z_abs = abs((np.log(max(float(val), 1e-3)) - target_mean) / target_std)
                    tail_weight = 1.0 + 0.25 * min(z_abs, 2.5)
                    batch_weights.append(inv_weights[bin_idx] * tail_weight)
                batch_weights = torch.tensor(batch_weights, dtype=torch.float32, device=device)

                optimizer.zero_grad()
                day_logits, pred_z, residual_z = model(imgs, patch_masks)
                reg_loss = weighted_regression_loss(pred_z, y_true_z, weights=batch_weights, beta=0.5, mse_weight=0.35)
                cls_loss = F.cross_entropy(day_logits, y_day, weight=class_weights_t, label_smoothing=0.05)
                residual_loss = (residual_z ** 2).mean()
                loss = reg_loss + cls_loss_weight * cls_loss + residual_l2_weight * residual_loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                train_loss_sum += loss.item()
                train_reg_sum += reg_loss.item()
                train_cls_sum += cls_loss.item()
                train_correct += (day_logits.argmax(dim=1) == y_day).sum().item()
                train_total += y_day.numel()

            scheduler.step()
            if epoch + 1 >= ema_start_epoch:
                ema_state = update_ema_state(ema_state, model, decay=0.97)

            model.eval()
            val_reg_sum = 0.0
            val_cls_sum = 0.0
            val_correct = 0
            val_total = 0
            with torch.no_grad():
                for imgs, patch_masks, y_day, modulus, _, _ in val_loader:
                    imgs = imgs.to(device)
                    patch_masks = patch_masks.to(device)
                    y_day = y_day.to(device)
                    y_value = modulus.to(device).float()
                    y_true_z = ((torch.log(y_value.clamp_min(1e-3)) - mean_t) / std_t).view(-1)
                    day_logits, pred_z, _ = model(imgs, patch_masks)
                    val_reg_sum += weighted_regression_loss(pred_z, y_true_z, beta=0.5, mse_weight=0.35).item()
                    val_cls_sum += F.cross_entropy(day_logits, y_day, weight=class_weights_t).item()
                    val_correct += (day_logits.argmax(dim=1) == y_day).sum().item()
                    val_total += y_day.numel()

            train_loss = train_loss_sum / len(train_loader)
            train_reg_loss = train_reg_sum / len(train_loader)
            train_cls_loss = train_cls_sum / len(train_loader)
            val_reg_loss = val_reg_sum / len(val_loader)
            val_cls_loss = val_cls_sum / len(val_loader)
            val_loss = val_reg_loss + cls_loss_weight * val_cls_loss
            training_history.append(
                {
                    "fold": fold,
                    "epoch": epoch + 1,
                    "train_loss": train_loss,
                    "train_reg_loss": train_reg_loss,
                    "train_cls_loss": train_cls_loss,
                    "val_loss": val_loss,
                    "val_reg_loss": val_reg_loss,
                    "val_cls_loss": val_cls_loss,
                    "train_day_acc": train_correct / max(train_total, 1),
                    "val_day_acc": val_correct / max(val_total, 1),
                    "lr": optimizer.param_groups[0]["lr"],
                }
            )
            pd.DataFrame(training_history).to_csv(history_csv_path, index=False)

            if (epoch + 1) % 5 == 0 or epoch == 0:
                lr = optimizer.param_groups[0]["lr"]
                print(
                    f"  Epoch {epoch + 1:03d} | LR: {lr:.1e} | "
                    f"Train Loss: {train_loss:.4f} "
                    f"(Reg {train_reg_loss:.4f}, Cls {train_cls_loss:.4f}) | "
                    f"Train Day Acc: {train_correct / max(train_total, 1):.4f} | "
                    f"Val Reg: {val_reg_loss:.4f} | "
                    f"Val Cls: {val_cls_loss:.4f} | "
                    f"Val Day Acc: {val_correct / max(val_total, 1):.4f}"
                )

            if epoch == epochs - 1:
                torch.save(ema_state if ema_state is not None else model.state_dict(), final_model_path)
                print(f"  --> Epoch {epochs} finished; saved final EMA weights for fold evaluation.")

        model.load_state_dict(torch.load(final_model_path, weights_only=True))

        # Fit both calibrators using this outer fold's training cells only.
        # Inference uses deterministic preprocessing and the same horizontal-
        # flip test-time augmentation (TTA) used for held-out validation cells.
        calibration_loader = DataLoader(
            Subset(clone_dataset_view(dataset, is_train=False), train_idx),
            batch_size=batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
        )
        train_cal_preds, train_cal_probs = [], []
        model.eval()
        with torch.no_grad():
            for imgs, patch_masks, _, _, _, _ in calibration_loader:
                imgs = imgs.to(device)
                patch_masks = patch_masks.to(device)
                batch_pred_z, batch_probs = [], []
                for flip in (False, True):
                    imgs_in = torch.flip(imgs, dims=[3]) if flip else imgs
                    day_logits, pred_z, _ = model(imgs_in, patch_masks)
                    batch_pred_z.append(pred_z)
                    batch_probs.append(torch.softmax(day_logits, dim=1))

                avg_pred_z = torch.stack(batch_pred_z).mean(dim=0)
                avg_probs = torch.stack(batch_probs).mean(dim=0)
                train_preds = torch.exp(avg_pred_z * std_t + mean_t)
                train_cal_preds.extend(
                    train_preds.clamp(1e-2, 1200.0).cpu().numpy()
                )
                train_cal_probs.extend(avg_probs.cpu().numpy())

        train_cal_preds = np.asarray(train_cal_preds)
        train_cal_probs = np.asarray(train_cal_probs)
        bias_corrector = LinearRegression().fit(
            train_cal_preds.reshape(-1, 1), targets[train_idx]
        )
        day_aware_corrector = LinearRegression().fit(
            np.column_stack([train_cal_preds, train_cal_probs]),
            targets[train_idx],
        )
        fold_records.append(
            {
                "fold": fold,
                "model_path": final_model_path,
                "state_dict": {name: param.detach().cpu() for name, param in model.state_dict().items()},
                "target_mean": target_mean,
                "target_std": target_std,
                "day_mean_z": [float(value) for value in day_mean_z],
                "train_indices": train_idx.tolist(),
                "val_indices": val_idx.tolist(),
                "bias_corrector": bias_corrector,
                "day_aware_bias_corrector": day_aware_corrector,
            }
        )

        model.eval()
        with torch.no_grad():
            for imgs, patch_masks, y_day, modulus, fnames, days in val_loader:
                imgs = imgs.to(device)
                patch_masks = patch_masks.to(device)
                batch_pred_z = []
                batch_probs = []
                for flip in [False, True]:
                    imgs_in = torch.flip(imgs, dims=[3]) if flip else imgs
                    day_logits, pred_z, _ = model(imgs_in, patch_masks)
                    batch_pred_z.append(pred_z.unsqueeze(0))
                    batch_probs.append(torch.softmax(day_logits, dim=1).unsqueeze(0))
                avg_pred_z = torch.cat(batch_pred_z, dim=0).mean(dim=0)
                avg_probs = torch.cat(batch_probs, dim=0).mean(dim=0)
                final_preds = torch.exp(avg_pred_z * std_t + mean_t).cpu().numpy().flatten()
                final_preds = np.clip(final_preds, 1e-2, 1200.0)
                pred_day = avg_probs.argmax(dim=1).cpu().numpy()
                probs_np = avg_probs.cpu().numpy()
                calibrated_preds = np.maximum(
                    bias_corrector.predict(final_preds.reshape(-1, 1)),
                    1e-2,
                )
                day_aware_calibrated_preds = np.maximum(
                    day_aware_corrector.predict(
                        np.column_stack([final_preds, probs_np])
                    ),
                    1e-2,
                )

                oof_reg_preds.extend(final_preds)
                oof_targets.extend(modulus.numpy())
                oof_day_true.extend(y_day.numpy())
                oof_day_pred.extend(pred_day)
                oof_day_prob.extend(probs_np)
                oof_calibrated_preds.extend(calibrated_preds)
                oof_day_aware_calibrated_preds.extend(
                    day_aware_calibrated_preds
                )
                oof_fnames.extend(fnames)
                oof_days.extend(days.numpy())
                oof_folds.extend([fold + 1] * len(fnames))

    targets_arr = np.array(oof_targets)
    preds_arr = np.array(oof_reg_preds)
    day_true_arr = np.array(oof_day_true)
    day_pred_arr = np.array(oof_day_pred)
    day_prob_arr = np.array(oof_day_prob)

    corrected_preds_arr = np.asarray(oof_calibrated_preds)
    day_aware_corrected_preds_arr = np.asarray(
        oof_day_aware_calibrated_preds
    )
    if not (
        len(corrected_preds_arr)
        == len(targets_arr)
        == len(day_aware_corrected_preds_arr)
    ):
        raise RuntimeError("OOF calibration output lengths are inconsistent.")

    # These full-OOF calibrators are deployment artifacts only. Reported OOF
    # metrics below use the fold-specific calibrators fitted above.
    final_bias_corrector = LinearRegression()
    final_bias_corrector.fit(preds_arr.reshape(-1, 1), targets_arr)

    day_cal_features = np.column_stack([preds_arr, day_prob_arr])
    final_day_aware_corrector = LinearRegression()
    final_day_aware_corrector.fit(day_cal_features, targets_arr)

    r2 = r2_score(targets_arr, preds_arr)
    mae = mean_absolute_error(targets_arr, preds_arr)
    rmse = np.sqrt(mean_squared_error(targets_arr, preds_arr))
    mape = mean_absolute_percentage_error(targets_arr, np.maximum(preds_arr, 1e-3))
    cal_r2 = r2_score(targets_arr, corrected_preds_arr)
    cal_mae = mean_absolute_error(targets_arr, corrected_preds_arr)
    cal_rmse = np.sqrt(mean_squared_error(targets_arr, corrected_preds_arr))
    cal_mape = mean_absolute_percentage_error(targets_arr, np.maximum(corrected_preds_arr, 1e-3))
    day_cal_r2 = r2_score(targets_arr, day_aware_corrected_preds_arr)
    day_cal_mae = mean_absolute_error(targets_arr, day_aware_corrected_preds_arr)
    day_cal_rmse = np.sqrt(mean_squared_error(targets_arr, day_aware_corrected_preds_arr))
    day_cal_mape = mean_absolute_percentage_error(targets_arr, np.maximum(day_aware_corrected_preds_arr, 1e-3))
    relative_errors = np.abs((targets_arr - preds_arr) / targets_arr)
    corrected_relative_errors = np.abs((targets_arr - corrected_preds_arr) / targets_arr)
    day_aware_corrected_relative_errors = np.abs((targets_arr - day_aware_corrected_preds_arr) / targets_arr)
    median_re = np.median(relative_errors)
    cal_median_re = np.median(corrected_relative_errors)
    day_cal_median_re = np.median(day_aware_corrected_relative_errors)
    pred_corr = np.corrcoef(targets_arr, preds_arr)[0, 1]
    day_cal_corr = np.corrcoef(targets_arr, day_aware_corrected_preds_arr)[0, 1]

    day_acc = accuracy_score(day_true_arr, day_pred_arr)
    day_bal_acc = balanced_accuracy_score(day_true_arr, day_pred_arr)
    day_macro_f1 = f1_score(day_true_arr, day_pred_arr, average="macro")
    day_weighted_f1 = f1_score(day_true_arr, day_pred_arr, average="weighted")
    day_cm = confusion_matrix(day_true_arr, day_pred_arr, labels=np.arange(len(class_names)))

    print("\n" + "=" * 50)
    print(f"{args.folds}-fold OOF hybrid day + modulus evaluation")
    print("=" * 50)
    print(f"Modulus R2: {r2:.4f}")
    print(f"Modulus MAE: {mae:.4f}")
    print(f"Modulus RMSE: {rmse:.4f}")
    print(f"Modulus MAPE: {mape * 100:.2f}%")
    print(f"Modulus Median RE: {median_re * 100:.2f}%")
    print(f"Calibrated R2: {cal_r2:.4f}")
    print(f"Calibrated MAE: {cal_mae:.4f}")
    print(f"Calibrated RMSE: {cal_rmse:.4f}")
    print(f"Calibrated MAPE: {cal_mape * 100:.2f}%")
    print(f"Calibrated Median RE: {cal_median_re * 100:.2f}%")
    print(f"Day-aware Calibrated R2: {day_cal_r2:.4f}")
    print(f"Day-aware Calibrated MAE: {day_cal_mae:.4f}")
    print(f"Day-aware Calibrated RMSE: {day_cal_rmse:.4f}")
    print(f"Day-aware Calibrated MAPE: {day_cal_mape * 100:.2f}%")
    print(f"Day-aware Calibrated Median RE: {day_cal_median_re * 100:.2f}%")
    print(f"Day-aware Pearson Corr: {day_cal_corr:.4f}")
    print(f"Pearson Corr: {pred_corr:.4f}")
    print(f"Day Accuracy: {day_acc:.4f}")
    print(f"Day Balanced Accuracy: {day_bal_acc:.4f}")
    print(f"Day Macro F1: {day_macro_f1:.4f}")
    print(f"Day Weighted F1: {day_weighted_f1:.4f}")
    print(f"Day Confusion Matrix rows=true cols=pred, class order {class_names}:")
    print(day_cm)
    print("\nDay Classification Report:")
    print(classification_report(day_true_arr, day_pred_arr, target_names=class_names, digits=4))

    print("\n>>> OOF modulus relationship by true day group:")
    for idx, name in enumerate(class_names):
        mask = day_true_arr == idx
        if not np.any(mask):
            continue
        group_r2 = r2_score(targets_arr[mask], preds_arr[mask]) if mask.sum() > 1 else float("nan")
        print(
            f"    {name}: n={mask.sum()}, true_mean={targets_arr[mask].mean():.2f}, "
            f"pred_mean={preds_arr[mask].mean():.2f}, MAE={mean_absolute_error(targets_arr[mask], preds_arr[mask]):.2f}, "
            f"R2={group_r2:.4f}"
        )

    pred_day_groups = [class_names[i] for i in day_pred_arr]
    true_day_groups = [class_names[i] for i in day_true_arr]
    df_results = pd.DataFrame(
        {
            "Filename": oof_fnames,
            "Fold": oof_folds,
            "True_Day": oof_days,
            "True_Day_Group": true_day_groups,
            "Predicted_Day_Group": pred_day_groups,
            "True_Modulus": targets_arr,
            "Raw_Modulus_Prediction": preds_arr,
            "Corrected_Modulus_Prediction": corrected_preds_arr,
            "Day_Aware_Corrected_Modulus_Prediction": day_aware_corrected_preds_arr,
            "Raw_Relative_Error": relative_errors,
            "Raw_Relative_Error_Percentage": relative_errors * 100,
            "Linear_Calibrated_Relative_Error": corrected_relative_errors,
            "Linear_Calibrated_Relative_Error_Percentage": corrected_relative_errors * 100,
            "Day_Aware_Calibrated_Relative_Error": day_aware_corrected_relative_errors,
            "Day_Aware_Calibrated_Relative_Error_Percentage": day_aware_corrected_relative_errors * 100,
            **{f"Prob_{name}": day_prob_arr[:, idx] for idx, name in enumerate(class_names)},
        }
    )
    csv_path = os.path.join(output_dir, "hybrid_multitask_oof_results.csv")
    df_results.sort_values(by="Raw_Relative_Error", ascending=False).to_csv(csv_path, index=False)
    pd.DataFrame(training_history).to_csv(history_csv_path, index=False)

    package = {
        "model_class": "DINOv2HybridDayModulus",
        "run_config": run_config,
        "day_classes": HybridDayModulusDataset.DAY_CLASSES,
        "day_groups": HybridDayModulusDataset.DAY_GROUPS,
        "r_threshold": r_threshold,
        "target_canvas_size": dataset.target_canvas_size,
        "classification_loss_weight": cls_loss_weight,
        "residual_l2_weight": residual_l2_weight,
        "fold_models": fold_records,
        "bias_corrector": final_bias_corrector,
        "day_aware_bias_corrector": final_day_aware_corrector,
        "oof_modulus_predictions": preds_arr,
        "oof_corrected_modulus_predictions": corrected_preds_arr,
        "oof_day_aware_corrected_modulus_predictions": day_aware_corrected_preds_arr,
        "oof_modulus_targets": targets_arr,
        "oof_day_true": day_true_arr,
        "oof_day_pred": day_pred_arr,
        "oof_day_prob": day_prob_arr,
        "oof_filenames": oof_fnames,
        "oof_folds": oof_folds,
        "metrics": {
            "r2": float(r2),
            "mae": float(mae),
            "rmse": float(rmse),
            "mape": float(mape),
            "median_relative_error": float(median_re),
            "calibrated_r2": float(cal_r2),
            "calibrated_mae": float(cal_mae),
            "calibrated_rmse": float(cal_rmse),
            "calibrated_mape": float(cal_mape),
            "calibrated_median_relative_error": float(cal_median_re),
            "day_aware_calibrated_r2": float(day_cal_r2),
            "day_aware_calibrated_mae": float(day_cal_mae),
            "day_aware_calibrated_rmse": float(day_cal_rmse),
            "day_aware_calibrated_mape": float(day_cal_mape),
            "day_aware_calibrated_median_relative_error": float(day_cal_median_re),
            "day_accuracy": float(day_acc),
            "day_balanced_accuracy": float(day_bal_acc),
            "day_macro_f1": float(day_macro_f1),
            "day_weighted_f1": float(day_weighted_f1),
        },
    }
    package_path = os.path.join(output_dir, "oof_dinov2_hybrid_multitask_ensemble.pth")
    torch.save(package, package_path)
    print(f">>> Hybrid OOF result CSV saved to: {csv_path}")
    print(f">>> Hybrid training history CSV saved to: {history_csv_path}")
    print(f">>> Hybrid model package saved to: {package_path}")


if __name__ == "__main__":
    main()
