import cv2
import mediapipe as mp
import os


MODEL_FILE = 'face_landmarker.task'
CAM_ID = 0
DOT_COLOR = (0, 255, 0)  # neon green. looks cool
WINDOW_NAME = "ORACLE v1.01"

def load_detector():
    # detects 1 face at a time
    options = mp.tasks.vision.FaceLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=MODEL_FILE),
        num_faces=1
    )
    return mp.tasks.vision.FaceLandmarker.create_from_options(options)


def draw_landmarks(frame, landmarks):
    h, w, _ = frame.shape
    for face in landmarks:
        for point in face:
            x, y = int(point.x * w), int(point.y * h)
            cv2.circle(frame, (x, y), 1, DOT_COLOR, -1)


def main():
    print("Starting ORACLE....")

    detector = load_detector()

    cam = cv2.VideoCapture(CAM_ID, cv2.CAP_DSHOW)
    if not cam.isOpened():
        print("couldn't access the webcam. Check if it's connected.")
        return

    print("Camera is ready. press esc to exit.")

    while True:
        ok, frame = cam.read()
        if not ok:
            break

        # color pallate conversion. BGR to RGB
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))

        if result.face_landmarks:
            draw_landmarks(frame, result.face_landmarks)

        cv2.imshow(WINDOW_NAME, frame)

        # esc to exit
        if cv2.waitKey(1) & 0xFF == 27:  
            break

    cam.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()