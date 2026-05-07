import argparse
from collections import deque

import cv2
import mediapipe as mp
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


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
        self.classifier = nn.Linear(hidden_dim, vocab_size + 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.classifier(out)


def get_point(lm, idx, w, h):
    return np.array([lm[idx].x * w, lm[idx].y * h], dtype=np.float32)


def normalize(vec):
    n = np.linalg.norm(vec)
    return vec / n if n != 0 else vec


def extract_frame_feature(hand_landmarks, w, h):
    lm = hand_landmarks.landmark
    frame_data = []

    for i in range(21):
        frame_data.extend([lm[i].x, lm[i].y, lm[i].z])

    wrist = np.array([lm[0].x, lm[0].y, lm[0].z], dtype=np.float32)
    for i in range(21):
        rel = np.array([lm[i].x, lm[i].y, lm[i].z], dtype=np.float32) - wrist
        frame_data.extend(rel.tolist())

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


def greedy_decode(logits, vocab):
    blank_id = len(vocab)
    pred = logits.argmax(dim=-1).cpu().numpy().tolist()

    tokens = []
    last = -1
    for p in pred:
        if p != blank_id and p != last:
            tokens.append(vocab[p])
        last = p
    return " ".join(tokens)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", default="../checkpoints/ctc_lstm_best.pt", help="训练得到的 checkpoint")
    parser.add_argument("--camera", type=int, default=2, help="摄像头索引")
    parser.add_argument("--window", type=int, default=48, help="滑动窗口长度")
    parser.add_argument("--min_frames", type=int, default=12, help="最少多少帧后开始推理")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(args.ckpt, map_location=device)
    vocab = ckpt["vocab"]
    input_dim = int(ckpt["input_dim"])

    model = CTCLSTM(
        input_dim=input_dim,
        vocab_size=len(vocab),
        hidden_dim=int(ckpt["hidden_dim"]),
        num_layers=int(ckpt["num_layers"]),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    mp_hands = mp.solutions.hands
    mp_draw = mp.solutions.drawing_utils
    seq_buffer = deque(maxlen=args.window)
    latest_text = ""

    cap = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW)
    with mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.7) as hands:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            h, w, _ = frame.shape
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(image_rgb)

            if results.multi_hand_landmarks:
                hand_landmarks = results.multi_hand_landmarks[0]
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                feat = extract_frame_feature(hand_landmarks, w, h)
                seq_buffer.append(feat)

                if len(seq_buffer) >= args.min_frames:
                    x = torch.from_numpy(np.array(seq_buffer, dtype=np.float32)).unsqueeze(0).to(device)  # [1,T,F]
                    with torch.no_grad():
                        logits = model(x)  # [1,T,C]
                        text = greedy_decode(logits[0], vocab)
                    latest_text = text if text else latest_text

            cv2.putText(frame, f"Pred: {latest_text}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (20, 220, 20), 2)
            cv2.putText(
                frame,
                f"Frames: {len(seq_buffer)}/{args.window}",
                (10, 65),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 0),
                2,
            )
            cv2.imshow("CTC+LSTM Realtime Inference", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
                break
            if key == ord("c"):  # clear
                seq_buffer.clear()
                latest_text = ""

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
