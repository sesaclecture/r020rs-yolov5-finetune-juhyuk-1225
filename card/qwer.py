from PIL import Image
import cv2
import numpy as np

# 변환할 GIF 파일과 저장할 MP4 파일 이름
gif_path = 'foot.gif'
mp4_path = 'foot.mp4'

# GIF 열기
gif = Image.open(gif_path)

# MP4 설정을 위한 정보 추출
width, height = gif.size
fps = gif.info.get('duration', 100) / 1000  # 프레임 간 시간(ms)을 초 단위로 변환
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(mp4_path, fourcc, 1.0/fps, (width, height))

# GIF의 모든 프레임을 동영상으로 저장
for i in range(gif.n_frames):
    gif.seek(i)
    # PIL 이미지를 OpenCV 이미지(numpy array)로 변환
    frame = np.array(gif.convert('RGB'))
    # RGB -> BGR 변환 (OpenCV는 BGR 순서 사용)
    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    out.write(frame_bgr)

print(f"'{gif_path}' 파일을 '{mp4_path}' 파일로 변환 완료!")
out.release()