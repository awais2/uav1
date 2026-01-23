#!/usr/bin/env python3
import time
import cv2
from ultralytics import YOLO

MODEL_PATH = "./yolo11n_ncnn_model"   # folder created by export
CAM_INDEX = 0                         # try 0,1 if needed
OUT_PATH = "person_out.avi"

W, H = 640, 480
INFER_IMGSZ = 320                     # 320 is much faster than 640 on Pi4
CONF = 0.35
PERSON_CLASS = 0                      # COCO: person

def main():
    # Camera
    cap = cv2.VideoCapture(CAM_INDEX, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, H)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    # Ask camera for MJPG (often reduces CPU load)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))

    if not cap.isOpened():
        raise RuntimeError("Could not open camera. Try CAM_INDEX=1 or check permissions (/dev/video0).")

    # Video writer (MJPG is usually light on CPU)
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    writer = cv2.VideoWriter(OUT_PATH, fourcc, 20.0, (W, H))

    # Load NCNN model
    model = YOLO(MODEL_PATH)

    prev_t = time.perf_counter()
    fps_smooth = 0.0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        t0 = time.perf_counter()

        # Person-only inference
        results = model.predict(
            source=frame,
            imgsz=INFER_IMGSZ,
            conf=CONF,
            classes=[PERSON_CLASS],
            verbose=False,
        )
        annotated = results[0].plot()

        t1 = time.perf_counter()
        infer_fps = 1.0 / max(t1 - t0, 1e-6)

        # Overall loop FPS (smooth)
        now = time.perf_counter()
        loop_fps = 1.0 / max(now - prev_t, 1e-6)
        prev_t = now
        fps_smooth = (0.9 * fps_smooth) + (0.1 * loop_fps) if fps_smooth else loop_fps

        # Overlay text
        cv2.putText(annotated, f"Loop FPS: {fps_smooth:.2f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
        cv2.putText(annotated, f"Infer FPS: {infer_fps:.2f}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

        # Show + Save
        cv2.imshow("Person Detection (NCNN)", annotated)
        writer.write(annotated)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    writer.release()
    cap.release()
    cv2.destroyAllWindows()
    print(f"Saved: {OUT_PATH}")

if __name__ == "__main__":
    main()
