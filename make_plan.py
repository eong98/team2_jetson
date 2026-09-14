# -*- coding: utf-8 -*-
"""
make_plan.py

카메라 화면에서 매장 평면도(도면)를 자동으로 만들어낸다. 동시에 homography.json도 생성한다.

[원리 - 왜 도면을 자동으로 만들 수 있나]
카메라가 비스듬히 내려다본 바닥을, 위에서 내려다본 모습으로 '펴는' 것을 정사영 보정
(rectification)이라고 한다. 바닥은 평면이므로 호모그래피 하나로 정확히 펼 수 있고,
펴놓은 결과 이미지가 바로 그 매장의 평면도가 된다.

게다가 이 방식은 캘리브레이션이 따로 필요 없다. 바닥 네 모서리를 어디로 펼지 정하는 순간
"카메라의 이 점 = 도면의 이 점" 대응 4쌍이 확정되기 때문에, 호모그래피가 부산물로 같이 나온다.
즉 대응점을 따로 찍을 필요 없이 **네 점만 지정하면 도면과 변환행렬이 동시에 만들어진다.**

[한계 - 솔직하게]
1. 완전 무인(클릭 0회)은 어렵다. 바닥 네 모서리를 자동으로 찾으려면 바닥 타일 격자나 벽-바닥
   경계선이 뚜렷해야 하는데, 실제 매장은 진열대가 바닥을 가려서 검출이 자주 실패한다.
   --auto 옵션으로 시도는 해보되, 실패하면 네 점을 직접 지정하는 게 확실하다.
2. 만들어진 도면은 '바닥 사진을 편 것'이라 건축 도면처럼 깔끔하지 않다. 진열대는 위에서 본
   모습이 아니라 옆면이 늘어져 보인다(바닥 평면 밖에 있는 물체라서). 위치 표시용으로는 충분하지만,
   보기 좋은 도면이 필요하면 이걸 밑그림 삼아 다시 그리는 게 낫다.
3. 실제 치수(m)를 맞추려면 바닥의 실제 크기를 알려줘야 한다(--real). 모르면 비율만 맞고
   축척은 임의가 된다. 어차피 서버는 0~1 비율 좌표를 쓰므로 축척을 몰라도 동작에는 문제없다.

[사용 방법]
  # 1) 젯슨 카메라에서 프레임 한 장 캡처
  python3 make_plan.py --grab frame.jpg

  # 2) 그 이미지에서 바닥 네 모서리를 지정해 도면 + homography.json 생성
  #    순서: 좌상(안쪽 왼쪽) 우상(안쪽 오른쪽) 우하(앞쪽 오른쪽) 좌하(앞쪽 왼쪽)
  python3 make_plan.py --image frame.jpg --corners "237,397 1037,371 1248,662 88,682" --real 6x4

  # 3) 자동 검출 시도 (실패할 수 있음)
  python3 make_plan.py --image frame.jpg --auto --real 6x4

네 모서리 좌표는 캘리브레이터 웹페이지에서 카메라 이미지를 올리고 클릭하면 바로 읽을 수 있다.
"""

import argparse
import json
import os

import cv2
import numpy as np

CSI_PIPELINE = (
    "nvarguscamerasrc num-buffers=10 ! "
    "video/x-raw(memory:NVMM), width=1280, height=720, framerate=30/1 ! "
    "nvvidconv flip-method=0 ! "
    "video/x-raw, width=1280, height=720, format=BGRx ! "
    "videoconvert ! video/x-raw, format=BGR ! appsink"
)

PIXELS_PER_METER = 100      # 도면 1px = 1cm


# ---------------------------------------------------------------------------
def grab_frame(path):
    """젯슨 CSI 카메라에서 한 장 캡처해서 저장한다."""
    cap = cv2.VideoCapture(CSI_PIPELINE, cv2.CAP_GSTREAMER)
    if not cap.isOpened():
        raise RuntimeError("카메라를 열 수 없습니다. 워커가 이미 카메라를 쓰고 있으면 먼저 종료하세요.")
    ok, frame = None, None
    for _ in range(10):        # 초반 몇 장은 노출이 안 잡혀서 버린다
        ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        raise RuntimeError("프레임을 읽지 못했습니다.")
    cv2.imwrite(path, frame)
    print("캡처 저장: %s (%dx%d)" % (path, frame.shape[1], frame.shape[0]))
    return frame


def order_corners(pts):
    """네 점을 좌상 -> 우상 -> 우하 -> 좌하 순서로 정렬한다.

    사용자가 아무 순서로 찍어도 되게 하려는 것. 합(x+y)이 가장 작은 게 좌상,
    가장 큰 게 우하이고, 차(y-x)로 나머지 둘을 가른다.
    """
    pts = np.array(pts, dtype=np.float32)
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).ravel()
    return np.array([
        pts[np.argmin(s)],   # 좌상
        pts[np.argmin(d)],   # 우상
        pts[np.argmax(s)],   # 우하
        pts[np.argmax(d)],   # 좌하
    ], dtype=np.float32)


def auto_detect_floor(image):
    """바닥 영역의 사각형을 자동으로 찾아본다. 실패하면 None.

    방법: 블러 -> 엣지 -> 가장 큰 사각형 윤곽. 바닥이 넓게 트여 있고 경계가 뚜렷할 때만 통한다.
    진열대가 바닥을 많이 가리는 매장에서는 대부분 실패하므로, 어디까지나 보조 기능이다.
    """
    h, w = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 9, 75, 75)
    edges = cv2.Canny(gray, 40, 130)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=2)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best, best_area = None, 0.0
    for c in contours:
        area = cv2.contourArea(c)
        if area < w * h * 0.12:          # 화면의 12% 미만은 바닥으로 보기엔 너무 작다
            continue
        approx = cv2.approxPolyDP(c, 0.02 * cv2.arcLength(c, True), True)
        if len(approx) == 4 and cv2.isContourConvex(approx) and area > best_area:
            best, best_area = approx.reshape(4, 2), area

    if best is None:
        return None
    return order_corners(best)


def build_plan(image, corners, plan_w, plan_h):
    """네 모서리를 직사각형으로 펴서 평면도를 만들고, 그 변환행렬을 함께 돌려준다."""
    src = order_corners(corners)
    dst = np.array([[0, 0], [plan_w, 0], [plan_w, plan_h], [0, plan_h]], dtype=np.float32)
    matrix = cv2.getPerspectiveTransform(src, dst)
    plan = cv2.warpPerspective(image, matrix, (plan_w, plan_h))
    return plan, src, dst


def draw_grid(plan, ppm):
    """1m 간격 격자와 눈금을 그려서 도면처럼 보이게 한다."""
    h, w = plan.shape[:2]
    out = plan.copy()
    overlay = out.copy()
    for x in range(0, w, ppm):
        cv2.line(overlay, (x, 0), (x, h), (255, 255, 255), 1)
    for y in range(0, h, ppm):
        cv2.line(overlay, (0, y), (w, y), (255, 255, 255), 1)
    out = cv2.addWeighted(overlay, 0.28, out, 0.72, 0)
    cv2.rectangle(out, (0, 0), (w - 1, h - 1), (40, 40, 40), 3)
    cv2.putText(out, "1 grid = 1m", (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (20, 20, 20), 3)
    cv2.putText(out, "1 grid = 1m", (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (245, 245, 245), 1)
    return out


def save_config(path, cno, image_size, plan_size, src, dst):
    cfg = {
        "cno": cno,
        "image_size": [int(image_size[0]), int(image_size[1])],
        "plan_size": [int(plan_size[0]), int(plan_size[1])],
        # 서버 /api/shopmap/issue 가 0~1 비율만 받으므로 기본 정규화
        "normalize": True,
        "points": [
            {"camera": [round(float(s[0]), 1), round(float(s[1]), 1)],
             "plan": [round(float(d[0]), 1), round(float(d[1]), 1)]}
            for s, d in zip(src, dst)
        ],
    }
    with open(path, "w") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)
    return cfg


# ---------------------------------------------------------------------------
def parse_corners(text):
    pts = []
    for token in text.replace(",", " ").split():
        pts.append(float(token))
    if len(pts) != 8:
        raise ValueError("모서리는 x,y 4쌍(숫자 8개)이어야 합니다. 지금 %d개" % len(pts))
    return [(pts[i], pts[i + 1]) for i in range(0, 8, 2)]


def main():
    ap = argparse.ArgumentParser(description="카메라 화면에서 매장 도면 + homography.json 자동 생성")
    ap.add_argument("--grab", metavar="FILE", help="젯슨 카메라에서 프레임 한 장 캡처하고 종료")
    ap.add_argument("--image", help="입력 카메라 이미지")
    ap.add_argument("--corners", help='바닥 네 모서리 "x1,y1 x2,y2 x3,y3 x4,y4"')
    ap.add_argument("--auto", action="store_true", help="바닥 사각형 자동 검출 시도")
    ap.add_argument("--real", default="6x4", help="바닥 실제 크기 가로x세로 (미터). 기본 6x4")
    ap.add_argument("--cno", type=int, default=1, help="CCTV 번호")
    ap.add_argument("--out", default="plan.png", help="생성할 도면 파일명")
    ap.add_argument("--config", default="homography.json", help="생성할 설정 파일명")
    args = ap.parse_args()

    if args.grab:
        grab_frame(args.grab)
        return

    if not args.image:
        ap.error("--image 또는 --grab 중 하나가 필요합니다.")

    image = cv2.imread(args.image)
    if image is None:
        raise SystemExit("이미지를 읽을 수 없습니다: %s" % args.image)
    ih, iw = image.shape[:2]

    try:
        rw, rh = [float(v) for v in args.real.lower().split("x")]
    except Exception:
        raise SystemExit("--real 형식은 6x4 처럼 '가로x세로' 입니다.")
    plan_w, plan_h = int(rw * PIXELS_PER_METER), int(rh * PIXELS_PER_METER)

    corners = None
    if args.corners:
        corners = parse_corners(args.corners)
    elif args.auto:
        corners = auto_detect_floor(image)
        if corners is None:
            raise SystemExit(
                "바닥 사각형을 자동으로 찾지 못했습니다.\n"
                "진열대가 바닥을 가리면 흔히 실패합니다. --corners 로 네 점을 직접 지정하세요.\n"
                '예: --corners "237,397 1037,371 1248,662 88,682"')
        print("자동 검출된 바닥 모서리:")
        for p in corners:
            print("   (%.0f, %.0f)" % (p[0], p[1]))
        print("※ 결과 도면이 이상하면 이 점들이 잘못 잡힌 것이니 --corners 로 직접 지정하세요.")
    else:
        ap.error("--corners 또는 --auto 중 하나가 필요합니다.")

    plan, src, dst = build_plan(image, corners, plan_w, plan_h)
    plan = draw_grid(plan, PIXELS_PER_METER)
    cv2.imwrite(args.out, plan)

    cfg = save_config(args.config, args.cno, (iw, ih), (plan_w, plan_h), src, dst)

    print("")
    print("도면 생성       : %s  (%dx%d px, %.1fm x %.1fm)" % (args.out, plan_w, plan_h, rw, rh))
    print("설정 생성       : %s  (대응점 4쌍, 0~1 정규화)" % args.config)
    print("")
    print("이제 이 두 파일을 젯슨 ~/yolov5/ 에 두면 워커가 좌표를 함께 전송합니다.")
    print("도면 이미지는 서버의 POST /api/shopmap/generate 로 업로드하면 AI 도면이 만들어집니다.")


if __name__ == "__main__":
    main()
