from myfuncs import *
from tqdm import tqdm

video = 'files/sample2.mp4'

cap = cv2.VideoCapture(video)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
frame_fps = int(cap.get(cv2.CAP_PROP_FPS))

video_out = 'files/output4.avi'
out = cv2.VideoWriter(video_out, cv2.VideoWriter_fourcc('D', 'I', 'V', 'X'), frame_fps, (frame_width, frame_height))

print("Processing Video")
for i in tqdm(range(frame_count)):
    run, frame = cap.read()
    output, points = pose_detect(frame)
    out.write(output)
out.release()
cap.release()
print("Done processing video")

# play_video(video_out)
