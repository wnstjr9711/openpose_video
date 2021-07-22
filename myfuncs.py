from openpose import *


def pose_detect(frame):
    frame_height, frame_width, _ = frame.shape
    blob = cv2.dnn.blobFromImage(frame, 1.0 / 255, (368, 368), (0, 0, 0), swapRB=False, crop=False)
    net.setInput(blob)
    output = net.forward()
    points = []
    threshold = 0.1
    for i in range(18):
        probMap = output[0, i, :, :]
        minVal, prob, minLoc, point = cv2.minMaxLoc(probMap)
        x = (frame_width * point[0]) / output.shape[3]
        y = (frame_height * point[1]) / output.shape[2]

        # 키포인트 검출한 결과가 0.1보다 크면(검출한곳이 위 BODY_PARTS랑 맞는 부위면) points에 추가, 검출했는데 부위가 없으면 None으로
        if prob > threshold:
            cv2.circle(frame, (int(x), int(y)), 8, (0, 255, 255), thickness=-1, lineType=cv2.FILLED)
            cv2.putText(frame, "{}".format(i), (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2,
                        lineType=cv2.LINE_AA)
            points.append((int(x), int(y)))
        else:
            points.append(None)

    # 이미지에 생성된 Keypoint를 사전에 정한 규칙에 따라 연결하기
    for pair in POSE_PAIRS:
        part_a = pair[0]
        part_b = pair[1]
        if points[part_a] and points[part_b]:
            cv2.line(frame, points[part_a], points[part_b], (0, 255, 255), 2)
            cv2.circle(frame, points[part_a], 8, (0, 0, 255), thickness=-1, lineType=cv2.FILLED)
    return frame


def play_video(video):
    cap = cv2.VideoCapture(video)
    while cap.isOpened():
        run, frame = cap.read()
        if not run:
            break
        cv2.imshow('video', frame)
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
