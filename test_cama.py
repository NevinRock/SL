import cv2

available = []
for i in range(10):  # 探测 0~9
    cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
    if cap.isOpened():
        ret, _ = cap.read()
        if ret:
            available.append(i)
    cap.release()

print("可用摄像头索引:", available)