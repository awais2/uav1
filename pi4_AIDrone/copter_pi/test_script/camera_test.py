import cv2

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Cannot open camera at index 0")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Can't receive frame")
        break

    cv2.imshow("Camera 0", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
