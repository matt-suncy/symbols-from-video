import cv2
import os 

vid_path = '/home/jovyan/Documents/latplan-temporal-segmentation/videos/assembly_C10118_rgb.mp4'
vid_name = vid_path.split('/')[-1]
output_path = '/home/jovyan/Documents/latplan-temporal-segmentation/videos/frames'
vidcap = cv2.VideoCapture(vid_path)
success,image = vidcap.read()
count = 0
while success:
    save_path = os.path.join(output_path, vid_name, "{:010d}.jpg".format(count))
    cv2.imwrite(save_path, image)     # save frame as JPEG file      
    success,image = vidcap.read()
    # print('Read a new frame: ', success)
    count += 1