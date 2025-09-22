import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import torch
import cv2
import os

# TODO: 커스텀 모델 로드
yolov5_repo_path = "./yolov5"
model_path = os.path.abspath("./best.pt") # .pt 모델 경로

model = torch.hub.load(yolov5_repo_path, 'custom', path=model_path, source='local')

# TODO: Label 로드

# Video capture
video_path = "./ball.mp4" # 처리할 MP4 영상 파일 경로
cap = cv2.VideoCapture(video_path)
input_size = 640

# Loop for camera frames
while True:
    # Read frame (BGR to RGB)
    ret, frame = cap.read()
    # break the loop on error
    if not ret:
        break

    # 추론 실행 (BGR -> RGB)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # TODO: 추론 전 입력 크기 보정 (640x640)
    resized_frame = cv2.resize(rgb_frame, (input_size, input_size))
    results = model(resized_frame)

    # TODO: 카메라 입력의 크기(frame_h, frame_w)와 모델의 입력 크기(input_h, input_w) 구하기
    frame_h, frame_w, _ = frame.shape

    # Boudning box 그리기
    for i, obj in enumerate(results.xyxy[0]):
        # 인식결과를 표시하기 위한 좌표를 얻음
        x1_model, y1_model, x2_model, y2_model, conf, cls_idx = obj
        
        # TODO: 출력 바운딩박스 크기 조절
        x1 = int(x1_model * (frame_w / input_size))
        y1 = int(y1_model * (frame_h / input_size))
        x2 = int(x2_model * (frame_w / input_size))
        y2 = int(y2_model * (frame_h / input_size))
        
        cls = int(cls_idx)
        
        # TODO: 인식된 정확도(confidence)와 클래스를 label로 구성
        label = model.names[cls]
        
        # OpenCV를 이용해서 해당 좌표에 사각형과 text를 출력
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        print(f"Object {i}: {label} at [{x1}, {y1}, {x2}, {y2}]")

    # 화면 표시
    cv2.imshow("YOLOv5", frame)

    # 종료를 위한 key 처리
    key = cv2.waitKey(1) & 0xFF
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()