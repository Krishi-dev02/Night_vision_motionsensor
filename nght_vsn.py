import cv2
import numpy as np
import time
import threading

import winsound
def beep():
    winsound.Beep(2500, 400)          

last_beep_time = 0
BEEP_COOLDOWN = 1.2     

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Cannot open webcam")
    exit()

cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)


bg_subtractor = cv2.createBackgroundSubtractorMOG2(
    history=500,            # longer history = more stable background (less sensitive to slow changes)
    varThreshold=25,        # higher = less sensitive to noise/shadows (16–40 range)
    detectShadows=True      # helps ignore moving shadows / lighting changes
)

# Shadow value in mask is usually ~127 → we will remove them
bg_subtractor.setShadowValue(0)           # shadows become 0 (background)
bg_subtractor.setShadowThreshold(0.5)     # tighter shadow detection (0.3–0.6)

print("Accurate motion detection started (MOG2). Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Optional: slight resize + blur to reduce webcam noise
    frame = cv2.resize(frame, (640, 480))
    frame_blur = cv2.GaussianBlur(frame, (5, 5), 0)

    # ── Apply background subtraction (this is the accurate part) ───────
    fg_mask = bg_subtractor.apply(frame_blur)

    # ── Clean up the mask for better contour detection ──────────────────
    # Remove shadows (value ~127) and small noise
    _, fg_mask_clean = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)

    # Morphological cleanup – removes small noise blobs, connects broken parts
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    fg_mask_clean = cv2.morphologyEx(fg_mask_clean, cv2.MORPH_OPEN,  kernel, iterations=1)   # remove tiny noise
    fg_mask_clean = cv2.morphologyEx(fg_mask_clean, cv2.MORPH_CLOSE, kernel, iterations=2)   # fill small holes

    # ── Find moving objects (contours) ──────────────────────────────────
    contours, _ = cv2.findContours(fg_mask_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    motion_detected = False

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 600:                  # raise to 800–1500 if too many false positives
            continue

        motion_detected = True

        # Draw nice bounding box
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 140), 2)
        # Optional: label area size for tuning
        # cv2.putText(frame, f"{int(area)}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,140), 2)

    # ── Beep when significant motion is present (cooldown) ──────────────
    current_time = time.time()
    if motion_detected and (current_time - last_beep_time > BEEP_COOLDOWN):
        threading.Thread(target=beep, daemon=True).start()
        last_beep_time = current_time

    # ── Night vision style overlay (green tint on original) ─────────────
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_boost = cv2.convertScaleAbs(gray, alpha=1.5, beta=25)
    green_tint = np.zeros_like(frame)
    green_tint[:,:,1] = gray_boost
    glow = cv2.GaussianBlur(green_tint, (0,0), sigmaX=2)
    night_vision = cv2.addWeighted(green_tint, 1.0, glow, 0.35, 0)

    # Draw motion boxes on night-vision view too
    for contour in contours:
        if cv2.contourArea(contour) >= 600:
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(night_vision, (x, y), (x + w, y + h), (0, 255, 100), 2)

    # Show result
    cv2.imshow("Accurate Night Vision + Motion Alert", night_vision)

    # Optional debug windows – very helpful for tuning
    # cv2.imshow("Foreground Mask (clean)", fg_mask_clean)
    # cv2.imshow("Original with boxes", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()