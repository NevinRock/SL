import argparse
from collections import deque

import cv2
import mediapipe as mp
import numpy as np
import torch
import torch.nn as nn

SINGLE_HAND_FEATURE_DIM = 21 * 3 + 21 * 3 + 5 * 2


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
        return self.classifier(context)  # [B, C]


def get_point(lm, idx, w, h):
    return np.array([lm[idx].x * w, lm[idx].y * h], dtype=np.float32)


def normalize(vec):
    n = np.linalg.norm(vec)
    return vec / n if n != 0 else vec


def zscore(x: np.ndarray) -> np.ndarray:
    mean = x.mean()
    std = x.std() + 1e-6
    return (x - mean) / std


def extract_frame_feature(hand_landmarks, w, h):
    lm = hand_landmarks.landmark
    frame_data = []

    # absolute xyz
    for i in range(21):
        frame_data.extend([lm[i].x, lm[i].y, lm[i].z])

    # relative xyz to wrist
    wrist = np.array([lm[0].x, lm[0].y, lm[0].z], dtype=np.float32)
    for i in range(21):
        rel = np.array([lm[i].x, lm[i].y, lm[i].z], dtype=np.float32) - wrist
        frame_data.extend(rel.tolist())

    # 5 normalized finger direction vectors (2D)
    vectors = {
        "thumb": get_point(lm, 4, w, h) - get_point(lm, 2, w, h),
        "index": get_point(lm, 8, w, h) - get_point(lm, 5, w, h),
        "middle": get_point(lm, 12, w, h) - get_point(lm, 9, w, h),
        "ring": get_point(lm, 16, w, h) - get_point(lm, 13, w, h),
        "pinky": get_point(lm, 20, w, h) - get_point(lm, 17, w, h),
    }
    for v in vectors.values():
        frame_data.extend(normalize(v).tolist())

    return np.array(frame_data, dtype=np.float32)


def extract_two_hand_frame_feature(results, w, h):
    hands_with_label = []
    multi_handedness = results.multi_handedness or []
    multi_hand_landmarks = results.multi_hand_landmarks or []

    for idx, hand_landmarks in enumerate(multi_hand_landmarks):
        label = "Unknown"
        if idx < len(multi_handedness) and multi_handedness[idx].classification:
            label = multi_handedness[idx].classification[0].label
        wrist_x = hand_landmarks.landmark[0].x
        hands_with_label.append((label, wrist_x, hand_landmarks))

    # Keep offline extraction order: Left first, then Right.
    label_priority = {"Left": 0, "Right": 1}
    hands_with_label.sort(key=lambda x: (label_priority.get(x[0], 2), x[1]))

    frame_data = []
    for _, _, hand_landmarks in hands_with_label[:2]:
        frame_data.extend(extract_frame_feature(hand_landmarks, w, h))

    missing_hands = 2 - len(hands_with_label[:2])
    if missing_hands > 0:
        frame_data.extend([0.0] * (missing_hands * SINGLE_HAND_FEATURE_DIM))

    return np.array(frame_data, dtype=np.float32)


def adapt_feature_dim(feat: np.ndarray, target_dim: int) -> np.ndarray:
    """Auto-adapt feature dim to checkpoint input_dim by padding/truncation."""
    cur = int(feat.shape[0])
    if cur == target_dim:
        return feat
    if cur > target_dim:
        return feat[:target_dim]
    out = np.zeros((target_dim,), dtype=np.float32)
    out[:cur] = feat
    return out


def infer_hidden_num_layers(state_dict):
    hidden_dim = int(state_dict["lstm.weight_hh_l0"].shape[1])
    layer_ids = set()
    for key in state_dict:
        if key.startswith("lstm.weight_ih_l"):
            suffix = key.split("lstm.weight_ih_l", 1)[1]
            if suffix.isdigit():
                layer_ids.add(int(suffix))
    num_layers = max(layer_ids) + 1 if layer_ids else 1
    return hidden_dim, num_layers


def format_topk_line(probs: torch.Tensor, vocab, k=5):
    k = min(k, probs.numel())
    vals, idxs = torch.topk(probs, k=k, dim=0)
    return " | ".join([f"{vocab[int(i)]}:{float(p) * 100:.1f}%" for p, i in zip(vals, idxs)])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt",  default= "../checkpoints/lstm_attn_20.pt", help="attn_lstm checkpoint 路径")
    parser.add_argument("--camera", type=int, default=0, help="摄像头索引")
    parser.add_argument("--window", type=int, default=32, help="滑动窗口长度")
    parser.add_argument("--min_frames", type=int, default=12, help="最少多少帧后开始推理")
    parser.add_argument("--topk", type=int, default=5, help="显示 top-k")
    parser.add_argument("--smooth", type=int, default=6, help="概率平滑窗口长度")
    parser.add_argument("--conf", type=float, default=0.55, help="显示主类别的最小置信度")
    parser.add_argument("--sentence_max_words", type=int, default=20, help="句子缓冲区最大词数")
    parser.add_argument("--stable_frames", type=int, default=3, help="同一词连续命中多少帧后写入句子")
    parser.add_argument("--cooldown_frames", type=int, default=8, help="写入一个词后最少间隔帧数")
    parser.add_argument(
        "--show_when_no_hand",
        action="store_true",
        help="丢手后继续显示上次预测（默认丢手后清空）",
    )
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(args.ckpt, map_location=device)
    if "model" not in ckpt or "vocab" not in ckpt or "input_dim" not in ckpt:
        raise ValueError("checkpoint 缺少必要字段，期望包含: model, vocab, input_dim")

    vocab = ckpt["vocab"]
    input_dim = int(ckpt["input_dim"])
    state_dict = ckpt["model"]
    hidden_dim, num_layers = infer_hidden_num_layers(state_dict)

    model = LSTMAttention(
        input_dim=input_dim,
        vocab_size=len(vocab),
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=0.3,
    ).to(device)
    model.load_state_dict(state_dict)
    model.eval()

    mp_hands = mp.solutions.hands
    mp_draw = mp.solutions.drawing_utils

    seq_buffer = deque(maxlen=args.window)
    prob_buffer = deque(maxlen=max(1, args.smooth))
    sentence_buffer = deque(maxlen=max(1, args.sentence_max_words))
    latest_text = ""
    latest_topk = ""
    latest_sentence = ""
    stable_pred_idx = -1
    stable_count = 0
    last_commit_idx = -1
    cooldown = 0

    cap = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise RuntimeError(f"无法打开摄像头: {args.camera}")

    with mp_hands.Hands(max_num_hands=2, model_complexity=0, min_detection_confidence=0.7, min_tracking_confidence=0.7) as hands:
        prev_base_feat = None
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            h, w, _ = frame.shape
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(image_rgb)
            has_hand = results.multi_hand_landmarks is not None

            if has_hand:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                base_feat = extract_two_hand_frame_feature(results, w, h)
                base_feat = zscore(base_feat)
                if prev_base_feat is None:
                    velocity = np.zeros_like(base_feat, dtype=np.float32)
                else:
                    velocity = base_feat - prev_base_feat
                velocity = zscore(velocity)
                velocity = np.clip(velocity, -3.0, 3.0)
                feat = np.concatenate([base_feat, velocity], axis=0).astype(np.float32)
                prev_base_feat = base_feat

                feat = adapt_feature_dim(feat, input_dim)
                seq_buffer.append(feat)

                if len(seq_buffer) >= args.min_frames:
                    x = torch.from_numpy(np.array(seq_buffer, dtype=np.float32)).unsqueeze(0).to(device)  # [1,T,F]
                    with torch.no_grad():
                        logits = model(x)[0]  # [C]
                        probs = torch.softmax(logits, dim=-1).detach().cpu()

                    prob_buffer.append(probs)
                    avg_probs = torch.stack(list(prob_buffer), dim=0).mean(dim=0)
                    pred_idx = int(torch.argmax(avg_probs).item())
                    pred_conf = float(avg_probs[pred_idx].item())

                    latest_topk = format_topk_line(avg_probs, vocab, args.topk)
                    if pred_conf >= args.conf:
                        latest_word = vocab[pred_idx]
                        latest_text = f"{latest_word} ({pred_conf * 100:.1f}%)"

                        if pred_idx == stable_pred_idx:
                            stable_count += 1
                        else:
                            stable_pred_idx = pred_idx
                            stable_count = 1

                        if cooldown > 0:
                            cooldown -= 1

                        # Sentence Buffer: 词在若干帧内稳定出现，且不在冷却期时写入句子
                        if (
                            stable_count >= args.stable_frames
                            and cooldown == 0
                            and pred_idx != last_commit_idx
                        ):
                            sentence_buffer.append(latest_word)
                            latest_sentence = " ".join(sentence_buffer)
                            last_commit_idx = pred_idx
                            cooldown = max(0, args.cooldown_frames)
                    else:
                        latest_text = f"... ({pred_conf * 100:.1f}%)"
                        stable_pred_idx = -1
                        stable_count = 0
            else:
                seq_buffer.clear()
                prob_buffer.clear()
                prev_base_feat = None
                stable_pred_idx = -1
                stable_count = 0
                if not args.show_when_no_hand:
                    latest_text = ""
                    latest_topk = ""

            cv2.putText(frame, f"Pred: {latest_text}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (20, 220, 20), 2)
            cv2.putText(
                frame,
                f"Frames: {len(seq_buffer)}/{args.window}",
                (10, 65),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 0),
                2,
            )
            if latest_topk:
                cv2.putText(frame, latest_topk, (10, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (240, 240, 240), 1)
            if latest_sentence:
                cv2.putText(
                    frame,
                    f"Sentence: {latest_sentence}",
                    (10, 125),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (20, 200, 255),
                    2,
                )

            cv2.imshow("Attention LSTM Realtime Inference", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
                break
            if key == ord("c"):  # clear
                seq_buffer.clear()
                prob_buffer.clear()
                sentence_buffer.clear()
                prev_base_feat = None
                latest_text = ""
                latest_topk = ""
                latest_sentence = ""
                stable_pred_idx = -1
                stable_count = 0
                last_commit_idx = -1
                cooldown = 0

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
