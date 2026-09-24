from pathlib import Path
import sys

try:
    import cv2
except ImportError:
    print("Error: OpenCV is not installed. Install it with `pip install opencv-python`.")
    sys.exit(1)


def display_video_information(video_path: str) -> None:
    """
    Read a video file, print its properties, and play it frame by frame.

    Controls:
        Space: Pause or resume
        Q/Esc: Stop playback
    """

    video_path = video_path.strip().strip('"').strip("'")
    path = Path(video_path).expanduser()

    if not path.is_file():
        print(f"Error: Video file not found: {path}")
        return

    video = cv2.VideoCapture(str(path))
    if not video.isOpened():
        print("Error: OpenCV could not open the video file.")
        return

    try:
        video_name = path.name
        frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(video.get(cv2.CAP_PROP_FPS))

        duration_seconds = total_frames / fps if fps > 0 else 0
        duration_minutes = int(duration_seconds // 60)
        remaining_seconds = duration_seconds % 60

        print("\n========== VIDEO PROPERTIES ==========")
        print(f"Video Name       : {video_name}")
        print(f"Resolution       : {frame_width} x {frame_height} pixels")
        print(f"Number of Frames : {total_frames}")
        print(f"Frame Rate       : {fps:.2f} FPS")
        print(f"Duration         : {duration_seconds:.2f} seconds")
        print(
            f"Duration Format  : "
            f"{duration_minutes:02d}:{remaining_seconds:05.2f}"
        )
        print("======================================")

        frame_delay = max(1, round(1000 / fps)) if fps > 0 else 30
        frame_number = 0
        paused = False

        print("\nControls")
        print("Press SPACE to pause or resume.")
        print("Press Q or ESC to stop.")

        while True:
            if not paused:
                success, frame = video.read()
                if not success:
                    print("\nEnd of video reached.")
                    break

                frame_number += 1
                frame_label = f"Frame: {frame_number}/{total_frames}"

                cv2.putText(
                    frame,
                    frame_label,
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )

                cv2.imshow("AI-Driven Video Processing", frame)

            delay = 30 if paused else frame_delay
            key = cv2.waitKey(delay) & 0xFF

            if key in (ord("q"), 27):
                print("\nVideo stopped by the user.")
                break

            if key == ord(" "):
                paused = not paused
    finally:
        video.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    file_path = input("Enter the complete path of the video file: ")
    display_video_information(file_path)