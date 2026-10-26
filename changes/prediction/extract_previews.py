import cv2

cap = cv2.VideoCapture("changes/prediction/outputs/viedo 1_unet_overlay.mp4")
targets = {100, 300, 600}
idx = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    if idx in targets:
        cv2.imwrite(f"changes/prediction/outputs/preview_frame_{idx}.png", frame)
    idx += 1

cap.release()
print(f"Extracted {len(targets)} preview frames!")
