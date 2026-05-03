import cv2
import mediapipe as mp
import os
import math
from collections import deque
import time
#----------------------------------------------------
MODEL_FILE = 'face_landmarker.task'
CAM_ID = 0
DOT_COLOR = (0, 255, 0)  # neon green. looks cool
WINDOW_NAME = "ORACLE v1.01"
CAL_FRAMES = 90

left_eye = [33, 160, 158, 133, 153, 144]
right_eye = [362, 385, 387, 263, 373, 380]

detect_blink = 0.25
i_cls_fr = 2
detect_suspision = 70
#------------------------------------------------------
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


def calc_eye_open(landmarks, eyes_indicators):
    points = [landmarks[i] for i in eyes_indicators]
    v1 = math.dist((points[1].x, points[1].y), (points[5].x, points[5].y))
    v2 = math.dist((points[2].x, points[2].y), (points[4].x, points[4].y))
    h  = math.dist((points[0].x, points[0].y), (points[3].x, points[3].y))
    if h == 0:
        return 0.3
    
    return (v1 + v2) / (2.0 * h)    # final ear

def calc_gaze(landmarks):
    l_iris = landmarks[468]
    r_iris = landmarks[473]
    l_corn = landmarks[33]
    r_corn = landmarks[263]

    l_gaze = l_iris.x - l_corn.x
    r_gaze = r_iris.x - r_corn.x

    gaze = (l_gaze + r_gaze) / 2    # "-" = looking right "+" = looking left
    return gaze


def main():
    print("Starting ORACLE....")
    detector = load_detector()

    cam = cv2.VideoCapture(CAM_ID, cv2.CAP_DSHOW)
    if not cam.isOpened():
        print("couldn't access the webcam. Check if it's connected.")
        return

    print("Camera is ready. press esc to exit.")

    blink_count = 0
    lst_avg_ear = 1.0   # last frames eye aspect ratio
    baseline_ear = None
    cal_frames = 0
    suspision_score = 0
    frame_history = 0     # last 30 frames
    i_cls_cnt = 0
    gaze_dur = 0
    sus_score = 0
    blink_times = deque()

    strt_time = time.time()     # blink rate calc

    while True:
        ok, frame = cam.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)

        # color cov from BGR to RGB
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))

        if result.face_landmarks:
            draw_landmarks(frame, result.face_landmarks)

            lnmk = result.face_landmarks[0]
            left_ear = calc_eye_open(lnmk, left_eye)
            right_ear = calc_eye_open(lnmk, right_eye)
            avg_ear = (left_ear + right_ear) / 2

            # MAHORAGA
            if cal_frames < CAL_FRAMES:
                if baseline_ear is None:
                    baseline_ear = avg_ear
                else:
                    baseline_ear = (baseline_ear * cal_frames + avg_ear) / (cal_frames + 1)
                cal_frames += 1
            detect_blink = baseline_ear * 0.74 if baseline_ear else 0.25    # multiplier 70-80

            if avg_ear < detect_blink:
                i_cls_cnt += 1
            else:
                if i_cls_cnt >= i_cls_fr:
                    blink_count += 1
                i_cls_cnt = 0

            lst_avg_ear = avg_ear

            # increases if look away. decays if not
            gaze = calc_gaze(lnmk)
            if abs(gaze) > 0.04:
                gaze_dur += 1
            else:
                gaze_dur = max(gaze_dur - 1,0)

            # blink rate vs norm. too high or too low is sus. norm = 0.25/sec
            elapsed = time.time() - strt_time
            blink_rate = blink_count / elapsed if elapsed > 0 else 0

            if cal_frames >= CAL_FRAMES:
                gaze_score = min(gaze_dur * 0.5, 60)
                blink_score = min(abs(blink_rate - 0.25) * 200, 40)
                target = blink_score + gaze_score
                sus_score = sus_score * 0.9 + target * 0.1   # keeps smooth
                

            h, w, _ = frame.shape
            cv2.putText(frame, f"Gaze: {gaze:.3f}", (w - 180, 110),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
            cv2.putText(frame, f"EAR: {avg_ear:.3f}", (w - 180, 50),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
            cv2.putText(frame, f"Blinks: {blink_count}", (w - 180, 80),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)  
            cv2.putText(frame, f"Suspicion: {int(sus_score)}%", (10,50),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255) if sus_score > detect_suspision else (0,255,0), 2)      

        cv2.imshow(WINDOW_NAME, frame)

        # esc to exit
        if cv2.waitKey(1) & 0xFF == 27:  
            break

    cam.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()