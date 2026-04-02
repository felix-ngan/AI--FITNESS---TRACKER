import cv2
import mediapipe as mp
import numpy as np
import joblib

# Charger modèle IA
model = joblib.load("exercise_model.pkl")

# MediaPipe
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_draw = mp.solutions.drawing_utils

# Calcul d'angle entre 3 points
def calculate_angle(a, b, c):
    a = np.array(a); b = np.array(b); c = np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    return 360-angle if angle > 180.0 else angle

cap = cv2.VideoCapture(0)
counter = 0
stage = None
current_exercise = "unknown"

while cap.isOpened():
    ret, frame = cap.read()
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)
    image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

    if results.pose_landmarks:
        mp_draw.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        lm = results.pose_landmarks.landmark

        # Raccourcis indices
        LS = mp_pose.PoseLandmark.LEFT_SHOULDER.value
        RS = mp_pose.PoseLandmark.RIGHT_SHOULDER.value
        LE = mp_pose.PoseLandmark.LEFT_ELBOW.value
        LW = mp_pose.PoseLandmark.LEFT_WRIST.value
        LH = mp_pose.PoseLandmark.LEFT_HIP.value
        LK = mp_pose.PoseLandmark.LEFT_KNEE.value
        LA = mp_pose.PoseLandmark.LEFT_ANKLE.value

        # Features (mêmes colonnes que dataset)
        features = [
            lm[LS].x, lm[LS].y, lm[RS].x, lm[RS].y,
            lm[LE].x, lm[LE].y, lm[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x, lm[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y,
            lm[LW].x, lm[LW].y, lm[mp_pose.PoseLandmark.RIGHT_WRIST.value].x, lm[mp_pose.PoseLandmark.RIGHT_WRIST.value].y,
            lm[LH].x, lm[LH].y, lm[mp_pose.PoseLandmark.RIGHT_HIP.value].x, lm[mp_pose.PoseLandmark.RIGHT_HIP.value].y,
            lm[LK].x, lm[LK].y, lm[mp_pose.PoseLandmark.RIGHT_KNEE.value].x, lm[mp_pose.PoseLandmark.RIGHT_KNEE.value].y
        ]

        # Prédiction IA
        current_exercise = model.predict([features])[0]

        # Comptage spécifique par exercice
        if current_exercise == "squat":
            hip = [lm[LH].x, lm[LH].y]
            knee = [lm[LK].x, lm[LK].y]
            ankle = [lm[LA].x, lm[LA].y]
            angle = calculate_angle(hip, knee, ankle)
            if angle < 70: stage = "down"
            if angle > 160 and stage == "down":
                stage = "up"; counter += 1

        elif current_exercise == "pushup":
            shoulder = [lm[LS].x, lm[LS].y]
            elbow = [lm[LE].x, lm[LE].y]
            wrist = [lm[LW].x, lm[LW].y]
            angle = calculate_angle(shoulder, elbow, wrist)
            if angle < 70: stage = "down"
            if angle > 160 and stage == "down":
                stage = "up"; counter += 1

        elif current_exercise == "situp":
            shoulder = [lm[LS].x, lm[LS].y]
            hip = [lm[LH].x, lm[LH].y]
            knee = [lm[LK].x, lm[LK].y]
            angle = calculate_angle(shoulder, hip, knee)
            if angle < 100: stage = "up"
            if angle > 160 and stage == "up":
                stage = "down"; counter += 1

        # Affichage
        cv2.putText(image, f'Exercice: {current_exercise}', (10,40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,0,0), 2)
        cv2.putText(image, f'Reps: {counter}', (10,80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)
        if stage:
            cv2.putText(image, f'Stage: {stage}', (10,120),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,200,255), 2)

    cv2.imshow('Projet Tutoré - IA Exercise Counter', image)
    key = cv2.waitKey(10) & 0xFF
    if key == ord('q'):
        break
    if key == ord('r'):
        counter = 0; stage = None  # reset compteur

cap.release()
cv2.destroyAllWindows()
