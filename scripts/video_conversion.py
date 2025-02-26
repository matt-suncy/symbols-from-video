#!/usr/bin/env python3
import subprocess
import os
import argparse

def convert_mp4_to_avi(input_path: str, output_path: str) -> None:
    """
    Convert an MP4 video to an AVI video using FFmpeg.

    Parameters:
        input_path (str): Path to the input MP4 video.
        output_path (str): Path where the output AVI video will be saved.
    """
    # Ensure the input file exists
    if not os.path.isfile(input_path):
        raise FileNotFoundError(f"Input file '{input_path}' does not exist.")

    # Construct the FFmpeg command.
    # This command uses the libxvid codec for video compression.
    command = [
        "ffmpeg",
        "-i", input_path,
        "-c:v", "libxvid",
        "-qscale:v", "3",  # quality level; lower is better quality
        output_path
    ]

    print(f"Converting '{input_path}' to '{output_path}'...")
    # Run the FFmpeg command and wait for it to complete.
    subprocess.run(command, check=True)
    print("Conversion completed successfully.")

def main():
    # parser = argparse.ArgumentParser(
    #     description="Convert an MP4 video to AVI format using FFmpeg."
    # )
    # parser.add_argument(
    #     "--input", "-i",
    #     type=str,
    #     default="input_video.mp4",
    #     help="Path to the input MP4 video file."
    # )
    # parser.add_argument(
    #     "--output", "-o",
    #     type=str,
    #     default="output_video.avi",
    #     help="Path to the output AVI video file."
    # )
    # args = parser.parse_args()

    input_path = '/home/jovyan/Documents/latplan-temporal-segmentation/videos/assembly_C10118_rgb.mp4'
    output_path = '/home/jovyan/Documents/latplan-temporal-segmentation/videos/assembly_C10118_rgb.avi'

    try:
        convert_mp4_to_avi(input_path, output_path)
    except subprocess.CalledProcessError as e:
        print(f"Error during conversion: {e}")
    except FileNotFoundError as e:
        print(e)

if __name__ == "__main__":
    main()
