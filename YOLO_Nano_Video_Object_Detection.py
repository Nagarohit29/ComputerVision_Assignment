import cv2
from ultralytics import YOLO
from pathlib import Path

VIDEO_PATH = "Simple Street Scene.mp4"          
MODEL_PATH = "yolo11n.pt"              
OUTPUT_DIR = Path("object_detection_outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

OUTPUT_VIDEO_PATH = OUTPUT_DIR / "detected_output.mp4"
CONFIDENCE_THRESHOLD = 0.40            
WINDOW_NAME = "YOLO Object Detection - Press Q to Quit"


def main():
    print(f"Loading model: {MODEL_PATH}")
    model = YOLO(MODEL_PATH)

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        raise FileNotFoundError(
            f"Could not open video: {VIDEO_PATH}\n"
            "Place the video in the same folder as this script or update VIDEO_PATH."
        )

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if fps <= 0:
        fps = 25.0

    print("\n--- Input Video Details ---")
    print(f"Resolution: {width} x {height}")
    print(f"FPS: {fps:.2f}")
    print(f"Total frames: {frame_count}")
    print(f"Confidence threshold: {CONFIDENCE_THRESHOLD:.2f}")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(
        str(OUTPUT_VIDEO_PATH), fourcc, fps, (width, height)
    )
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(
            "Could not create the output video. Try using an AVI output with XVID codec."
        )

    frame_index = 0
    total_detections = 0
    frames_with_detections = 0

    try:
        while True:
            success, frame = cap.read()
            if not success:
                break

            frame_index += 1

            results = model.predict(
                source=frame,
                conf=CONFIDENCE_THRESHOLD,
                imgsz=640,
                verbose=False
            )

            result = results[0]
            annotated_frame = result.plot()

            detections_in_frame = 0
            if result.boxes is not None:
                detections_in_frame = len(result.boxes)
            total_detections += detections_in_frame
            if detections_in_frame > 0:
                frames_with_detections += 1

            writer.write(annotated_frame)

            cv2.imshow(WINDOW_NAME, annotated_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == 27:
                print("Stopped by user.")
                break

            if frame_index % 30 == 0:
                print(
                    f"Processed {frame_index}"
                    + (f"/{frame_count}" if frame_count > 0 else "")
                    + f" frames | detections in current frame: {detections_in_frame}"
                )

    finally:
        cap.release()
        writer.release()
        cv2.destroyAllWindows()

    print("\n--- Object Detection Summary ---")
    print(f"Frames processed: {frame_index}")
    print(f"Frames containing detections: {frames_with_detections}")
    print(f"Total detections across all processed frames: {total_detections}")
    print(f"Saved annotated video: {OUTPUT_VIDEO_PATH.resolve()}")
    print("Note: total detections counts each detected object in each frame; "
          "it is not the number of unique objects.")

if __name__ == "__main__":
    main()