from myfuncs import *

video = 'files/sample.mp4'
cap = cv2.VideoCapture(video)

frame_width, frame_height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

video_out = 'files/output2.avi'
out = cv2.VideoWriter(video_out, cv2.VideoWriter_fourcc('D','I','V','X'), fps, (frame_width, frame_height))

print("Processing Video...")
while cap.isOpened():
    run, frame = cap.read()
    if not run:
        break
    output = pose_detect(frame)
    out.write(output)
out.release()
cap.release()
print("Done processing video")

play_video(video_out)
