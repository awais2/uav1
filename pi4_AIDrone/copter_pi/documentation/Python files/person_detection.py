#!/usr/bin/env python
    model_path = "yolo11n.pt"
    out_path = Path("person_output_640x480.mp4")

    conf = 0.25
    iou = 0.45
    # COCO "person" class id is 0
    person_class_id = 0

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

    # Read one frame to confirm actual size
    ok, frame = cap.read()
    if not ok:
        raise RuntimeError("Could not read from camera.")
    h, w = frame.shape[:2]
    print(f"[INFO] Camera stream: {w}x{h}")

    # -----------------------------
    # Video writer
    # -----------------------------
    # Try to use camera FPS; if unavailable, fallback
    cam_fps = cap.get(cv2.CAP_PROP_FPS)
    if cam_fps is None or cam_fps <= 1:
        cam_fps = 30.0

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, cam_fps, (w, h))
    if not writer.isOpened():
        raise RuntimeError("Could not open VideoWriter. Try installing ffmpeg or use .avi + XVID.")

    # -----
