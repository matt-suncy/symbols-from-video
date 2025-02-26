import cv2
import os
import sys
import argparse

def extract_frames(video_path: str, output_folder: str) -> None:
    """
    Extracts frames from the given video file and saves them as JPEG images in the output folder.
    
    Parameters:
        video_path (str): The path to the video file.
        output_folder (str): The directory where the frames will be saved.
    """
    # Create the output folder if it doesn't exist.
    os.makedirs(output_folder, exist_ok=True)
    
    # Attempt to open the video file.
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file '{video_path}'.")
        sys.exit(1)
    
    frame_count = 0
    while True:
        # Read a frame from the video.
        ret, frame = cap.read()
        if not ret:
            # If no frame is returned, we've reached the end of the video.
            break
        
        # Create a filename for the frame.
        frame_filename = os.path.join(output_folder, f"frame_{frame_count:05d}.jpg")
        
        # Attempt to write the frame to file.
        try:
            cv2.imwrite(frame_filename, frame)
        except Exception as e:
            print(f"Error writing frame {frame_count}: {e}")
        frame_count += 1

    # Release the video capture object.
    cap.release()
    print(f"Extracted {frame_count} frames to '{output_folder}'.")

def main():
    # parser = argparse.ArgumentParser(description="Extract frames from a video file.")
    # parser.add_argument("video_path", help="Path to the input video file")
    # parser.add_argument("output_folder", help="Folder where the extracted frames will be saved")
    # args = parser.parse_args()

    # try:
    #     extract_frames(args.video_path, args.output_folder)
    # except Exception as e:
    #     print(f"An unexpected error occurred: {e}")
    #     sys.exit(1)

    video_path = '/home/jovyan/Documents/latplan-temporal-segmentation/videos/assembly_C10118_rgb.mp4'
    output_folder = '/home/jovyan/Documents/latplan-temporal-segmentation/videos/frames/assembly_C10118_rgb'

    try:
        extract_frames(video_path=video_path, output_folder=output_folder)
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
