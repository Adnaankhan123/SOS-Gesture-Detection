import cv2
import mediapipe as mp
import time

def main():
    # Open the default camera (webcam)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open video stream")
        return

    # Try to get the FPS from the camera; default to 25 if unavailable.
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 25

    # Initialize MediaPipe Hands.
    mp_hands = mp.solutions.hands
    hands_detector = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    mp_drawing = mp.solutions.drawing_utils

    # Variables to track closed-hand gestures and SOS detection.
    closed_gesture_count = 0  # Count of distinct closed-hand events.
    prev_closed = False       # Hand state in the previous frame.
    sos_detect_time = None     # Timestamp when SOS was triggered.
    current_bounds = None      # Bounding box of the detected closed hand.

    while True:
        start_time = time.time()
        ret, frame = cap.read()
        if not ret:
            print("Warning: Unable to read frame from the camera")
            break

        image_height, image_width, _ = frame.shape

        # Convert the image from BGR to RGB for MediaPipe.
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands_detector.process(frame_rgb)

        # For each frame check if a closed hand is detected.
        current_closed = False
        detected_bounds = None
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Convert landmarks to pixel coordinates.
                landmarks_px = [
                    (int(lm.x * image_width), int(lm.y * image_height))
                    for lm in hand_landmarks.landmark
                ]
                xs = [pt[0] for pt in landmarks_px]
                ys = [pt[1] for pt in landmarks_px]
                x_min, x_max = min(xs), max(xs)
                y_min, y_max = min(ys), max(ys)

                # Heuristic for a closed hand: All four fingers folded.
                folded = 0
                finger_indices = [(8, 6), (12, 10), (16, 14), (20, 18)]
                for tip_idx, pip_idx in finger_indices:
                    if hand_landmarks.landmark[tip_idx].y > hand_landmarks.landmark[pip_idx].y:
                        folded += 1

                # If all four fingers are folded, consider this as a closed hand.
                if folded == 4:
                    current_closed = True
                    detected_bounds = (x_min, y_min, x_max, y_max)
                    # Process only the first detected closed hand.
                    break

        # Detect transition from open to closed: a distinct event.
        if current_closed and not prev_closed:
            closed_gesture_count += 1
            current_bounds = detected_bounds

        prev_closed = current_closed

        # When the second distinct closed-hand event occurs, trigger SOS.
        if closed_gesture_count == 2 and sos_detect_time is None:
            sos_detect_time = time.time()
            print("SOS gesture triggered.")

        # If SOS is active, draw the bounding box for 1.5 seconds.
        if sos_detect_time is not None:
            elapsed_since_sos = time.time() - sos_detect_time
            if elapsed_since_sos < 1.5:
                cv2.rectangle(frame, (current_bounds[0], current_bounds[1]),
                              (current_bounds[2], current_bounds[3]), (0, 0, 255), 2)
                cv2.putText(frame, "SOS Detected", (current_bounds[0], current_bounds[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            # After 2 seconds, reset the gesture detection state.
            if elapsed_since_sos >= 2.0:
                sos_detect_time = None
                closed_gesture_count = 0

        # Optionally, draw all detected hand landmarks (visual feedback).
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # Show the processed frame.
        cv2.imshow("SOS Gesture Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # Maintain original speed of the video.
        elapsed = time.time() - start_time
        sleep_duration = max(0, (1.0 / fps) - elapsed)
        time.sleep(sleep_duration)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()