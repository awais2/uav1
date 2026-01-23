#!/usr/bin/env python3
#python thermal_person_detect.py
import cv2
import time
from pathlib import Path
from ultralytics import YOLO


def main():
    # -----------------------------
    # Config
    # -----------------------------
    model_path = "./yolo11n_ncnn_model"  # change if needed
    out_path = Path("person_output_640x480.mp4")

    cam_index = 0
    width = 640
    height = 480

    conf = 0.25
    iou = 0.45
    person_class_id = 0  # COCO "person" class id

    # -----------------------------
    # Load model
    # -----------------------------
    model = YOLO(model_path)

    # -----------------------------
    # Camera
    # -----------------------------
    cap = cv2.VideoCapture(cam_index)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera index {cam_index}. Try cam_index=1")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    ok, frame = cap.read()
    if not ok:
        raise RuntimeError("Could not read from camera.")
    h, w = frame.shape[:2]
    print(f"[INFO] Camera stream: {w}x{h}")

    # -----------------------------
    # Video writer
    # -----------------------------
    cam_fps = cap.get(cv2.CAP_PROP_FPS)
    if cam_fps is None or cam_fps <= 1:
        cam_fps = 30.0

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, cam_fps, (w, h))
    if not writer.isOpened():
        raise RuntimeError("Could not open VideoWriter. Try installing ffmpeg or use .avi + XVID.")

    print("[INFO] Press 'q' to quit.")

    # -----------------------------
    # FPS helpers
    # -----------------------------
    prev_t = time.perf_counter()
    fps_ema = 0.0
    alpha = 0.15  # smoothing factor (0.1-0.2 is nice)

    # -----------------------------
    # Loop
    # -----------------------------
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("[WARN] Frame grab failed, stopping.")
                break

            # Inference
            results = model.predict(
                source=frame,
                conf=conf,
                iou=iou,
                classes=[person_class_id],
                verbose=False
            )

            # Draw detections
            annotated = results[0].plot() if results and len(results) > 0 else frame

            # FPS calc (overall pipeline FPS)
            now = time.perf_counter()
            dt = now - prev_t
            prev_t = now
            inst_fps = (1.0 / dt) if dt > 0 else 0.0
            fps_ema = inst_fps if fps_ema == 0.0 else (alpha * inst_fps + (1 - alpha) * fps_ema)

            # Overlay FPS
            fps_text = f"FPS: {fps_ema:.1f}"
            cv2.putText(
                annotated, fps_text, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA
            )

            # Show + save
            cv2.imshow("Person Detection", annotated)
            writer.write(annotated)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        cap.release()
        writer.release()
        cv2.destroyAllWindows()
        print(f"[INFO] Saved output video: {out_path.resolve()}")


if __name__ == "__main__":
    main()
