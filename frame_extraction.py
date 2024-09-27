import cv2
vidcap = cv2.VideoCapture('videos/chinchess_gettyimages-148739276-640_adpp.mp4')
success,image = vidcap.read()
count = 0
while success:
  cv2.imwrite("videos/chinchess_frames/frame%d.jpg" % count, image)     # save frame as JPEG file      
  success,image = vidcap.read()
  print('Read a new frame: ', success)
  count += 1