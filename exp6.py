from pathlib import Path
import sys

import cv2
import numpy as np


video_path = (
    sys.argv[1]
    if len(sys.argv) > 1
    else str(Path(__file__).resolve().parent / "exp31.mp4")
)

cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print(f"Error: Could not open video: {video_path}")
    sys.exit(1)


width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

print("========================================")
print("        VIDEO PROPERTIES")
print("========================================")
print("Video        :", video_path)
print("Resolution   :", width, "x", height)
print("FPS          :", fps)
print("Total Frames :", total_frames)

if fps > 0:
    duration = total_frames / fps
    print("Duration     :", round(duration, 2), "seconds")



background_subtractor = cv2.createBackgroundSubtractorMOG2(
    history=500,
    varThreshold=50,
    detectShadows=True
)



interval = 20
frame_number = 0


min_area = 500



while True:

    ret, frame = cap.read()

    if not ret:
        break

    
    if frame_number % interval == 0:

        print("\n----------------------------------------")
        print("Frame Number:", frame_number)



        foreground_mask = background_subtractor.apply(frame)

        
        _, threshold = cv2.threshold(
            foreground_mask,
            200,
            255,
            cv2.THRESH_BINARY
        )

        
        kernel = np.ones((5, 5), np.uint8)

        threshold = cv2.morphologyEx(
            threshold,
            cv2.MORPH_OPEN,
            kernel
        )

        threshold = cv2.dilate(
            threshold,
            kernel,
            iterations=2
        )



        contours, _ = cv2.findContours(
            threshold,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        object_count = 0

        for contour in contours:

            area = cv2.contourArea(contour)

            
            if area < min_area:
                continue

            x, y, w, h = cv2.boundingRect(contour)

            object_count += 1

            
            cv2.rectangle(
                frame,
                (x, y),
                (x + w, y + h),
                (0, 255, 0),
                2
            )

            
            cv2.putText(
                frame,
                f"Object {object_count}",
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

 

        print("Number of Objects Detected:", object_count)



        cv2.putText(
            frame,
            f"Objects Detected: {object_count}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2
        )

        cv2.putText(
            frame,
            f"Frame: {frame_number}",
            (20, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 0, 0),
            2
        )



        cv2.imshow(
            "Traffic Video - Object Detection",
            frame
        )

        
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    frame_number += 1



cap.release()
cv2.destroyAllWindows()

print("\n========================================")
print("Video processing completed successfully.")
print("========================================")
