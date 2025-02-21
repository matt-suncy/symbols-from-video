#!/usr/bin/env python3

import sys

def parse_line(line):
    """
    Parse a single annotation line and extract:
      - The video base (first three '-' separated parts).
      - The starting frame number (from the 6th '-' separated part, e.g., "F006979").
    Returns a tuple (video_base, start_frame, line) for sorting.
    """
    line = line.strip()
    if not line:
        return None
    # Split line into tokens (first token is the video name)
    tokens = line.split()
    video_name = tokens[0]
    # Split the video name by '-' assuming the format:
    # Participant - Session - Activity - <num> - <num> - F<start_frame> - F<end_frame>
    parts = video_name.split('-')
    if len(parts) < 7:
        print(f"Unexpected video name format: {video_name}")
        return None
    video_base = '-'.join(parts[:3])
    # Extract the starting frame from parts[5] (e.g., "F006979")
    try:
        start_frame = int(parts[5][1:])  # remove the 'F' and convert to int
    except ValueError:
        print(f"Error parsing frame number from {parts[5]} in {video_name}")
        return None
    return (video_base, start_frame, line)

def main():
    if len(sys.argv) < 2:
        print("Usage: python sort_annotations.py <input_file>")
        sys.exit(1)
    
    input_file = sys.argv[1]
    with open(input_file, 'r') as f:
        lines = f.readlines()
    
    parsed_lines = []
    for line in lines:
        parsed = parse_line(line)
        if parsed is not None:
            parsed_lines.append(parsed)
    
    # Sort first by video base name (alphabetically) then by the starting frame number.
    parsed_lines.sort(key=lambda x: (x[0], x[1]))
    
    # Output the sorted annotations.
    for _, _, original_line in parsed_lines:
        print(original_line.strip())

if __name__ == '__main__':
    main()
