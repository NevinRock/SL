import argparse
import os
import random
from collections import Counter
from datetime import datetime
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, Dataset
try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_VECTOR_BASE = os.path.normpath(
    os.path.join(BASE_DIR, "..", "..", "Data", "ASL_Citizen", "Dataset_500","dataset_split")
)


# =========================
# Dataset + Augmentation
# =========================
class VectorDataset(Dataset):
    def __init__(self, samples: List[Tuple[str, int]], augment=False):
        self.samples = samples
        self.augment = augment

    def __len__(self):
        return len(self.samples)

    def augment_feat(self, seq):
        # ===== 数据增强 =====
        if np.random.rand() < 0.5:
            noise = np.random.normal(0, 0.01, seq.shape)
            seq = seq + noise

        # 时间mask（类似SpecAugment）
        if np.random.rand() < 0.3:
            T = seq.shape[0]
            t = np.random.randint(0, T)
            w = np.random.randint(1, max(2, T // 10))
            seq[t:t + w] = 0

        return seq

    def __getitem__(self, idx):
        npy_path, label = self.samples[idx]
        seq = np.load(npy_path).astype(np.float32)

        if self.augment:
            seq = self.augment_feat(seq)

        return torch.from_numpy(seq), label


# =========================
# Collate
# =========================
def collate_fn(batch):
    seqs, labels = zip(*batch)
    lengths = [s.size(0) for s in seqs]
    max_len = max(lengths)
    feat_dim = seqs[0].size(1)

    padded = torch.zeros(len(seqs), max_len, feat_dim)

    for i, s in enumerate(seqs):
        padded[i, :s.size(0)] = s

    return padded, torch.tensor(labels)


# =========================
# LSTM + Attention Model
# =========================
class LSTMAttention(nn.Module):
    def __init__(self, input_dim, vocab_size, hidden_dim=256, num_layers=2, dropout=0.3):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=False,
        )

        self.hidden_dim = hidden_dim
        self.attn = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.Tanh(),
            nn.Linear(self.hidden_dim, 1),
        )

        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.hidden_dim, vocab_size)

    def forward(self, x):
        out, _ = self.lstm(x)  # [B, T, H]

        attn = torch.softmax(self.attn(out), dim=1)  # [B, T, 1]
        context = (out * attn).sum(dim=1)  # [B, H]

        context = self.dropout(context)
        logits = self.classifier(context)
        return logits


# =========================
# Utils
# =========================
def build_samples(vector_root: str, class_to_idx=None):
    if not os.path.isdir(vector_root):
        raise ValueError(f"目录不存在: {vector_root}")

    class_names = sorted([d for d in os.listdir(vector_root) if os.path.isdir(os.path.join(vector_root, d))])
    if not class_names:
        raise ValueError(f"未在 {vector_root} 找到类别子文件夹")

    if class_to_idx is None:
        class_to_idx = {name: i for i, name in enumerate(class_names)}
    unknown_class_count = 0

    samples = []
    for cls in class_names:
        if cls not in class_to_idx:
            unknown_class_count += 1
            continue
        cls_dir = os.path.join(vector_root, cls)
        for fn in os.listdir(cls_dir):
            if fn.endswith(".npy"):
                samples.append((os.path.join(cls_dir, fn), class_to_idx[cls]))

    if not samples:
        raise ValueError(f"未在 {vector_root} 找到任何可用 .npy 文件")
    if unknown_class_count > 0:
        print(f"[警告] {vector_root} 中有 {unknown_class_count} 个类别不在训练词表中，已跳过")

    return samples, class_to_idx


# =========================
# Train / Eval
# =========================
def run_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0

    iterator = loader
    if tqdm is not None:
        iterator = tqdm(loader, desc="Train", leave=False, ncols=100)

    for step, (x, y) in enumerate(iterator, start=1):
        x = x.to(device)
        y = y.to(device)

        logits = model(x)
        logits = logits / 0.7
        loss = criterion(logits, y)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()

        total_loss += loss.item()
        if tqdm is not None:
            iterator.set_postfix(loss=f"{(total_loss / step):.4f}")

    return total_loss / len(loader)


@torch.no_grad()
def eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct_top1 = 0
    correct_top5 = 0
    total = 0

    iterator = loader
    if tqdm is not None:
        iterator = tqdm(loader, desc="Eval ", leave=False, ncols=100)

    for step, (x, y) in enumerate(iterator, start=1):
        x = x.to(device)
        y = y.to(device)

        logits = model(x)
        logits = logits / 0.7
        loss = criterion(logits, y)

        total_loss += loss.item()

        pred_top1 = logits.argmax(dim=1)
        correct_top1 += (pred_top1 == y).sum().item()

        k = min(5, logits.size(1))
        pred_topk = logits.topk(k, dim=1).indices
        correct_top5 += (pred_topk == y.unsqueeze(1)).any(dim=1).sum().item()
        total += y.size(0)
        if tqdm is not None:
            iterator.set_postfix(loss=f"{(total_loss / step):.4f}")

    acc_top1 = correct_top1 / total
    acc_top5 = correct_top5 / total
    return total_loss / len(loader), acc_top1, acc_top5


# =========================
# Main
# =========================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="", help="YAML 配置文件路径")
    parser.add_argument("--vector_base", default=DEFAULT_VECTOR_BASE, help="包含 train/val/test 的向量根目录")
    parser.add_argument("--train_root", default="", help="训练集目录，默认使用 vector_base/train")
    parser.add_argument("--val_root", default="", help="验证集目录，默认使用 vector_base/val")
    parser.add_argument("--test_root", default="", help="测试集目录，默认使用 vector_base/test")
    parser.add_argument("--save_path", default="checkpoints/lstm_attn.pt")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--log_dir", default="runs/train_attn_lstm")
    parser.add_argument("--run_name", default="", help="本次训练的运行名；为空时自动用时间戳")
    parser.add_argument("--tb_new_run", action="store_true", help="为本次训练创建独立 TensorBoard 子目录")
    args = parser.parse_args()
    cfg = vars(args).copy()

    if args.config:
        with open(args.config, "r", encoding="utf-8") as f:
            yaml_cfg = yaml.safe_load(f) or {}
        cfg.update(yaml_cfg)

    train_root = cfg["train_root"] or os.path.join(cfg["vector_base"], "train")
    val_root = cfg["val_root"] or os.path.join(cfg["vector_base"], "val")
    test_root = cfg["test_root"] or os.path.join(cfg["vector_base"], "test")

    train_samples, class_to_idx = build_samples(train_root)
    val_samples, _ = build_samples(val_root, class_to_idx=class_to_idx)
    test_samples = []
    if os.path.isdir(test_root):
        test_samples, _ = build_samples(test_root, class_to_idx=class_to_idx)
    else:
        print(f"[提示] 未找到 test 目录，跳过测试评估: {test_root}")

    vocab = [None] * len(class_to_idx)
    for name, idx in class_to_idx.items():
        vocab[idx] = name

    random.shuffle(train_samples)

    input_dim = np.load(train_samples[0][0]).shape[1]

    train_ds = VectorDataset(train_samples, augment=True)
    val_ds = VectorDataset(val_samples, augment=False)
    test_ds = VectorDataset(test_samples, augment=False) if test_samples else None

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["batch_size"],
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=cfg.get("num_workers", 4),
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg["batch_size"],
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=cfg.get("num_workers", 4),
        pin_memory=torch.cuda.is_available(),
    )
    if test_ds is not None:
        test_loader = DataLoader(
            test_ds,
            batch_size=cfg["batch_size"],
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=cfg.get("num_workers", 4),
            pin_memory=torch.cuda.is_available(),
        )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    base_log_dir = cfg.get("log_dir", "runs/train_attn_lstm")
    run_name = cfg.get("run_name", "") or datetime.now().strftime("%Y%m%d-%H%M%S")
    use_new_run = bool(cfg.get("tb_new_run", False))
    actual_log_dir = os.path.join(base_log_dir, run_name) if use_new_run else base_log_dir
    writer = SummaryWriter(log_dir=actual_log_dir)

    print(
        f"device={device}, classes={len(vocab)}, train={len(train_ds)}, "
        f"val={len(val_ds)}, test={len(test_ds) if test_ds is not None else 0}, "
        f"input_dim={input_dim}, log_dir={actual_log_dir}"
    )

    model = LSTMAttention(
        input_dim=input_dim,
        vocab_size=len(vocab),
        hidden_dim=cfg["hidden_dim"],
        num_layers=cfg["num_layers"],
    ).to(device)

    labels = [label for _, label in train_samples]
    counter = Counter(labels)
    num_classes = len(class_to_idx)
    total = sum(counter.values())
    weights = []
    for i in range(num_classes):
        weights.append(total / (counter[i] + 1e-6))
    weights = torch.tensor(weights, dtype=torch.float32).to(device)
    weights = weights / weights.mean()

    criterion = nn.CrossEntropyLoss(
        weight=weights,
        label_smoothing=0.1,
    )

    # ⭐ weight decay = 正则化
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=1e-4)

    best_top1 = float("-inf")

    for epoch in range(1, cfg["epochs"] + 1):
        train_loss = run_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_top1, val_top5 = eval_epoch(model, val_loader, criterion, device)
        writer.add_scalar("loss/train", train_loss, epoch)
        writer.add_scalar("loss/val", val_loss, epoch)
        writer.add_scalar("acc_top1/val", val_top1, epoch)
        writer.add_scalar("acc_top5/val", val_top5, epoch)

        print(
            f"[{epoch:03d}/{cfg['epochs']}] train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
            f"top1={val_top1:.4f} top5={val_top5:.4f}"
        )
        if val_top1 > best_top1:
            best_top1 = val_top1
            torch.save(
                {
                    "model": model.state_dict(),
                    "vocab": vocab,
                    "input_dim": input_dim,
                    "best_val_top1": val_top1,
                    "best_val_top5": val_top5,
                    "best_val_loss": val_loss,
                },
                cfg["save_path"],
            )
            print("✅ saved best model")

    if test_loader is not None:
        test_loss, test_top1, test_top5 = eval_epoch(model, test_loader, criterion, device)
        print(f"[test] loss={test_loss:.4f} top1={test_top1:.4f} top5={test_top5:.4f}")
        writer.add_scalar("loss/test", test_loss, 0)
        writer.add_scalar("acc_top1/test", test_top1, 0)
        writer.add_scalar("acc_top5/test", test_top5, 0)

    writer.close()


if __name__ == "__main__":
    main()