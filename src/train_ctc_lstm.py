import argparse
import os
import random
from datetime import datetime
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter


class VectorCTCDataset(Dataset):
    def __init__(self, samples: List[Tuple[str, int]]):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        npy_path, label = self.samples[idx]
        seq = np.load(npy_path).astype(np.float32)  # [T, F]
        target = np.array([label], dtype=np.int64)  # isolated word -> one token
        return torch.from_numpy(seq), torch.from_numpy(target)


def ctc_collate(batch):
    seqs, targets = zip(*batch)
    lengths = torch.tensor([s.size(0) for s in seqs], dtype=torch.long)
    feat_dim = seqs[0].size(1)
    max_len = int(lengths.max().item())

    padded = torch.zeros(len(seqs), max_len, feat_dim, dtype=torch.float32)
    for i, s in enumerate(seqs):
        padded[i, : s.size(0)] = s

    target_lengths = torch.tensor([t.numel() for t in targets], dtype=torch.long)
    targets_concat = torch.cat(targets, dim=0)
    return padded, lengths, targets_concat, target_lengths


class CTCLSTM(nn.Module):
    def __init__(self, input_dim: int, vocab_size: int, hidden_dim: int = 256, num_layers: int = 2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=False,
            dropout=0.2 if num_layers > 1 else 0.0,
        )
        self.classifier = nn.Linear(hidden_dim, vocab_size + 1)  # +1 for CTC blank

    def forward(self, x):
        out, _ = self.lstm(x)
        logits = self.classifier(out)
        return logits


def build_samples(vector_root: str, class_to_idx=None):
    if not os.path.isdir(vector_root):
        raise ValueError(f"目录不存在: {vector_root}")

    class_names = sorted(
        [d for d in os.listdir(vector_root) if os.path.isdir(os.path.join(vector_root, d))]
    )
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


def run_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    for padded, input_lengths, targets_concat, target_lengths in loader:
        padded = padded.to(device)
        input_lengths = input_lengths.to(device)
        targets_concat = targets_concat.to(device)
        target_lengths = target_lengths.to(device)

        logits = model(padded)  # [B, T, C]
        log_probs = F.log_softmax(logits, dim=-1).permute(1, 0, 2)  # [T, B, C]

        loss = criterion(log_probs, targets_concat, input_lengths, target_lengths)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        total_loss += float(loss.item())

    return total_loss / max(1, len(loader))


@torch.no_grad()
def ctc_token_accuracy(logits, input_lengths, targets_concat, target_lengths, blank_idx):
    pred_ids = logits.argmax(dim=-1)  # [B, T]
    offset = 0
    correct = 0
    total = int(target_lengths.numel())
    for i in range(pred_ids.size(0)):
        T = int(input_lengths[i].item())
        seq = pred_ids[i, :T].tolist()

        decoded = []
        prev = None
        for token in seq:
            if token != blank_idx and token != prev:
                decoded.append(token)
            prev = token

        target_len = int(target_lengths[i].item())
        target_seq = targets_concat[offset : offset + target_len].tolist()
        offset += target_len

        if decoded == target_seq:
            correct += 1

    return correct / max(1, total)


@torch.no_grad()
def eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_acc = 0.0
    steps = 0
    blank_idx = criterion.blank
    for padded, input_lengths, targets_concat, target_lengths in loader:
        padded = padded.to(device)
        input_lengths = input_lengths.to(device)
        targets_concat = targets_concat.to(device)
        target_lengths = target_lengths.to(device)

        logits = model(padded)
        log_probs = F.log_softmax(logits, dim=-1).permute(1, 0, 2)
        loss = criterion(log_probs, targets_concat, input_lengths, target_lengths)
        total_loss += float(loss.item())
        total_acc += ctc_token_accuracy(logits, input_lengths, targets_concat, target_lengths, blank_idx)
        steps += 1

    return total_loss / max(1, len(loader)), total_acc / max(1, steps)


def main():
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config", default="", help="YAML 配置文件路径")
    config_args, remaining_argv = config_parser.parse_known_args()

    yaml_cfg = {}
    if config_args.config:
        with open(config_args.config, "r", encoding="utf-8") as f:
            yaml_cfg = yaml.safe_load(f) or {}
        if not isinstance(yaml_cfg, dict):
            raise ValueError("配置文件内容必须是键值对字典")

    parser = argparse.ArgumentParser(parents=[config_parser])
    parser.set_defaults(
        vector_base="../../Data/ASL_Citizen/Dataset_500/dataset_split",
        train_root="",
        val_root="",
        test_root="",
        save_path="checkpoints/ctc_lstm_best_500.pt",
        epochs=80,
        batch_size=16,
        hidden_dim=256,
        num_layers=2,
        lr=1e-3,
        seed=42,
        log_dir="runs/train_ctc_lstm",
        run_name="",
        tb_new_run=False,
    )
    parser.set_defaults(**yaml_cfg)
    parser.add_argument("--vector_base", help="包含 train/val/test 的向量根目录")
    parser.add_argument("--train_root", help="训练集目录，默认使用 vector_base/train")
    parser.add_argument("--val_root", help="验证集目录，默认使用 vector_base/val")
    parser.add_argument("--test_root", help="测试集目录，默认使用 vector_base/test")
    parser.add_argument("--save_path", help="模型保存路径")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--hidden_dim", type=int)
    parser.add_argument("--num_layers", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--log_dir")
    parser.add_argument("--run_name", help="本次训练的运行名；为空时自动用时间戳")
    parser.add_argument("--tb_new_run", action="store_true", help="为本次训练创建独立 TensorBoard 子目录")
    args = parser.parse_args(remaining_argv)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    train_root = args.train_root or os.path.join(args.vector_base, "train")
    val_root = args.val_root or os.path.join(args.vector_base, "val")
    test_root = args.test_root or os.path.join(args.vector_base, "test")

    train_samples, class_to_idx = build_samples(train_root)
    val_samples, _ = build_samples(val_root, class_to_idx=class_to_idx)
    test_samples, _ = build_samples(test_root, class_to_idx=class_to_idx)

    vocab = [None] * len(class_to_idx)
    for name, idx in class_to_idx.items():
        vocab[idx] = name

    input_dim = int(np.load(train_samples[0][0]).shape[1])
    train_ds = VectorCTCDataset(train_samples)
    val_ds = VectorCTCDataset(val_samples)
    test_ds = VectorCTCDataset(test_samples)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=ctc_collate)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=ctc_collate)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, collate_fn=ctc_collate)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CTCLSTM(
        input_dim=input_dim,
        vocab_size=len(vocab),
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
    ).to(device)

    criterion = nn.CTCLoss(blank=len(vocab), zero_infinity=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    run_name = args.run_name or datetime.now().strftime("%Y%m%d-%H%M%S")
    actual_log_dir = os.path.join(args.log_dir, run_name) if args.tb_new_run else args.log_dir
    writer = SummaryWriter(log_dir=actual_log_dir)

    save_dir = os.path.dirname(args.save_path)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    best_val = float("inf")

    print(
        f"device={device}, classes={vocab}, train={len(train_ds)}, "
        f"val={len(val_ds)}, test={len(test_ds)}, input_dim={input_dim}, log_dir={actual_log_dir}"
    )
    for epoch in range(1, args.epochs + 1):
        train_loss = run_epoch(model, train_loader, optimizer, criterion, device)
        _, train_acc = eval_epoch(model, train_loader, criterion, device)
        val_loss, val_acc = eval_epoch(model, val_loader, criterion, device)
        writer.add_scalar("loss/train", train_loss, epoch)
        writer.add_scalar("loss/val", val_loss, epoch)
        writer.add_scalar("acc/train", train_acc, epoch)
        writer.add_scalar("acc/val", val_acc, epoch)
        print(
            f"[{epoch:03d}/{args.epochs}] train_loss={train_loss:.4f} "
            f"train_acc={train_acc:.4f} val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )

        if val_loss < best_val:
            best_val = val_loss
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "vocab": vocab,
                    "input_dim": input_dim,
                    "hidden_dim": args.hidden_dim,
                    "num_layers": args.num_layers,
                },
                args.save_path,
            )
            print(f"  -> saved best checkpoint: {args.save_path}")

    test_loss, test_acc = eval_epoch(model, test_loader, criterion, device)
    print(f"[test] loss={test_loss:.4f} acc={test_acc:.4f}")
    writer.add_scalar("loss/test", test_loss, 0)
    writer.add_scalar("acc/test", test_acc, 0)
    writer.close()


if __name__ == "__main__":
    main()
