import cv2
import mediapipe as mp
import csv

# Init MediaPipe
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_draw = mp.solutions.drawing_utils

# Choix de l'exercice à enregistrer
exercise = input("Exercice à enregistrer (squat/pushup/situp): ").lower()

cap = cv2.VideoCapture(0)

# Colonnes: on enregistre 10 points (x,y) = 20 features + label
columns = [
    "ls_x","ls_y","rs_x","rs_y",
    "le_x","le_y","re_x","re_y",
    "lw_x","lw_y","rw_x","rw_y",
    "lh_x","lh_y","rh_x","rh_y",
    "lk_x","lk_y","rk_x","rk_y",
    "label"
]

# Crée le CSV si nécessaire et ajoute des lignes
with open("dataset.csv", "a", newline="") as f:
    writer = csv.writer(f)

    # Écrit l'entête si le fichier est vide
    try:
        f.seek(0)
        if f.read(1) == "":
            writer.writerow(columns)
    except:
        pass

    while cap.isOpened():
        ret, frame = cap.read()
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if results.pose_landmarks:
            mp_draw.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            lm = results.pose_landmarks.landmark

            # Raccourcis
            LS = mp_pose.PoseLandmark.LEFT_SHOULDER.value
            RS = mp_pose.PoseLandmark.RIGHT_SHOULDER.value
            LE = mp_pose.PoseLandmark.LEFT_ELBOW.value
            RE = mp_pose.PoseLandmark.RIGHT_ELBOW.value
            LW = mp_pose.PoseLandmark.LEFT_WRIST.value
            RW = mp_pose.PoseLandmark.RIGHT_WRIST.value
            LH = mp_pose.PoseLandmark.LEFT_HIP.value
            RH = mp_pose.PoseLandmark.RIGHT_HIP.value
            LK = mp_pose.PoseLandmark.LEFT_KNEE.value
            RK = mp_pose.PoseLandmark.RIGHT_KNEE.value

            row = [
                lm[LS].x, lm[LS].y, lm[RS].x, lm[RS].y,
                lm[LE].x, lm[LE].y, lm[RE].x, lm[RE].y,
                lm[LW].x, lm[LW].y, lm[RW].x, lm[RW].y,
                lm[LH].x, lm[LH].y, lm[RH].x, lm[RH].y,
                lm[LK].x, lm[LK].y, lm[RK].x, lm[RK].y,
                exercise
            ]
            writer.writerow(row)

        cv2.putText(image, f"Recording: {exercise} | q=quit", (10,30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
        cv2.imshow("Collecte de données", image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
