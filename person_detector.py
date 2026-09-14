# -*- coding: utf-8 -*-
"""
person_detector.py

worker_skeleton.py의 get_person_boxes()가 호출하는 실제 YOLOv5 추론 코드.
~/yolov5 (ultralytics/yolov5 v5.0 태그, yolov5s.pt) 안에 이 파일을 같이 두고 써야 한다
(models/, utils/ 패키지를 그대로 import하기 때문 - yolov5 저장소 밖으로 옮기면 임포트가 깨짐).

모델은 최초 호출 시 한 번만 로드해서 전역에 캐싱한다(매 프레임마다 새로 로드하면 절대 안 됨 -
나노에서 프레임마다 모델 로딩하면 너무 느려서 사실상 못 씀).

[2026-09-14 변경] IMG_SIZE를 640 -> 416으로 낮춤.
  이유: 젯슨 나노에서 640으로 돌리면 추론이 ~0.25초/프레임(=약 4fps)이라, 프레임 간 사람이
  너무 많이 움직여서 트래커가 같은 사람을 계속 다른 사람으로 인식한다(track ID churn).
  ID가 흔들리면 "같은 두 사람이 N초간 엉켜있다" 같은 폭행/체류 판정이 아예 성립을 못 한다.
  416으로 낮추면 연산량이 (416/640)^2 = 약 42%로 줄어 2배 이상 빨라지고, 사람처럼 큰 객체는
  정확도 손실이 거의 없다. 더 빠르게 하려면 320까지 낮출 수 있지만 멀리 있는 사람을 놓치기 시작한다.
  환경변수로도 바꿀 수 있게 해둠: YOLO_IMG_SIZE=320 python3 worker_skeleton.py
"""

import os

import numpy as np
import torch

from models.experimental import attempt_load
from utils.datasets import letterbox
from utils.general import non_max_suppression, scale_coords

WEIGHTS_PATH = "yolov5s.pt"   # ~/yolov5/yolov5s.pt (COCO pretrained)
IMG_SIZE = int(os.environ.get("YOLO_IMG_SIZE", 416))  # 640 -> 416 (속도 우선). 320까지 낮출 수 있음.
PERSON_CLASS_IDX = 0          # COCO 클래스 0번 = person
CONF_THRES = 0.4
IOU_THRES = 0.45

_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_model = None


def _load_model():
    global _model
    if _model is None:
        print("[person_detector] 모델 로딩 중... (device=%s, img_size=%d)" % (_device, IMG_SIZE))
        _model = attempt_load(WEIGHTS_PATH, map_location=_device)
        _model.eval()
        print("[person_detector] 모델 로딩 완료")
    return _model


def get_person_boxes(frame):
    """
    frame: cv2로 읽은 BGR 이미지 (numpy array)
    반환: [((x1, y1, x2, y2), det_confidence), ...]  - worker_skeleton.py가 기대하는 형식
    """
    model = _load_model()
    img0 = frame

    img = letterbox(img0, new_shape=IMG_SIZE)[0]
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR->RGB, HWC->CHW
    img = np.ascontiguousarray(img)

    img_t = torch.from_numpy(img).to(_device).float() / 255.0
    if img_t.ndimension() == 3:
        img_t = img_t.unsqueeze(0)

    with torch.no_grad():
        pred = model(img_t)[0]

    pred = non_max_suppression(pred, CONF_THRES, IOU_THRES, classes=[PERSON_CLASS_IDX])

    results = []
    for det in pred:
        if det is not None and len(det):
            det[:, :4] = scale_coords(img_t.shape[2:], det[:, :4], img0.shape).round()
            for *xyxy, conf, cls in det:
                x1, y1, x2, y2 = [float(v) for v in xyxy]
                results.append(((x1, y1, x2, y2), float(conf)))

    return results
