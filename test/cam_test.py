import cv2, time

cap = cv2.VideoCapture(2, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap.set(cv2.CAP_PROP_FPS, 30)

if not cap.isOpened():
    print("Cannot open camera")
    exit()

print("opened:", cap.isOpened(), "size:", cap.get(3), cap.get(4), "fps:", cap.get(5))

while True:
    ok, frame = cap.read()
    if not ok:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    cv2.imshow('Camera Preview', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
