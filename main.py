import cv2
import mediapipe as mp
import numpy as np


MP_POSE = mp.solutions.pose
MP_DRAWING = mp.solutions.drawing_utils.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2)
MP_DRAWING_CONNECTED = mp.solutions.drawing_utils.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
ANGLE_THRESHOLD_DOWN = 160 
ANGLE_THRESHOLD_UP = 30  
FONT = cv2.FONT_HERSHEY_SIMPLEX

def calculate_angle(a, b, c):
    """Calculates the angle between three points."""
    a = np.array(a)  # First point
    b = np.array(b)  # Mid point
    c = np.array(c)  # End point
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180.0:
        angle = 360 - angle
    return angle

def visualize_landmarks(image, results):
    """Renders body keypoints and connections on the frame."""
    mp.solutions.drawing_utils.draw_landmarks(image, results.pose_landmarks, MP_POSE.POSE_CONNECTIONS, MP_DRAWING, MP_DRAWING_CONNECTED)

def display_curl_counter(image, counter, stage):
    """Displays the current rep count and stage on the frame."""
    cv2.rectangle(image, (0, 0), (225, 73), (245, 117, 16), -1)  # Status box
    cv2.putText(image, 'REPS', (15, 12), FONT, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(image, str(counter), (10, 60), FONT, 2, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(image, 'STAGE', (65, 12), FONT, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(image, stage, (60, 60), FONT, 2, (255, 255, 255), 2, cv2.LINE_AA)



def get_user_feedback():
    """Prompts the user for feedback on the rep count."""
    while True:
        feedback = input("Enter 'y' if the rep count is accurate, or 'n' to adjust: ")
        if feedback.lower() in ('y', 'n'):
            return feedback.lower() == 'y'
        print("Invalid input. Please enter 'y' or 'n'.")

def main():
    cap = cv2.VideoCapture(0)
    counter = 0
    stage = None
    total_reps = 0  
    correct_reps = 0 

    with MP_POSE.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()

            if not ret:
                print("Error: Unable to capture frame from the webcam")
                break

            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = True
            results = pose.process(image)
            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            if results.pose_landmarks:
                # Extract keypoints
                shoulder = [results.pose_landmarks.landmark[MP_POSE.PoseLandmark.LEFT_SHOULDER].x,
                            results.pose_landmarks.landmark[MP_POSE.PoseLandmark.LEFT_SHOULDER].y]
                elbow = [results.pose_landmarks.landmark[MP_POSE.PoseLandmark.LEFT_ELBOW].x,
                         results.pose_landmarks.landmark[MP_POSE.PoseLandmark.LEFT_ELBOW].y]
                wrist = [results.pose_landmarks.landmark[MP_POSE.PoseLandmark.LEFT_WRIST].x,
                         results.pose_landmarks.landmark[MP_POSE.PoseLandmark.LEFT_WRIST].y]

                # Calculate curl angle
                angle = calculate_angle(shoulder, elbow, wrist)

                # Curl counter logic
                if angle > ANGLE_THRESHOLD_DOWN:
                    stage = "down"
                if angle < ANGLE_THRESHOLD_UP and stage == 'down':
                    stage = "up"
                    counter += 1
                    total_reps += 1
                    if get_user_feedback():
                        correct_reps += 1

                visualize_landmarks(image, results)
                display_curl_counter(image, counter, stage)

            cv2.imshow('Mediapipe Feed', image)

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

    if total_reps > 0:
        accuracy_percentage = (correct_reps / total_reps) * 100
        print(f"Accuracy: {accuracy_percentage:.2f}% ({correct_reps}/{total_reps} correct reps)")
    else:
        print("No reps detected.")

if __name__ == '__main__':
    main()
