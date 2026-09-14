# -*- coding: utf-8 -*-
"""
worker_skeleton.py

젯슨에서 돌아가는 메인 루프. 카메라(또는 테스트 영상 파일) -> 사람 탐지 -> 추적 -> 룰 판정
-> 서버로 이벤트 전송(비동기), + 디버그 화면 스트리밍.

이 파일은 ~/yolov5 (ultralytics/yolov5 v5.0 + yolov5s.pt) 폴더 안에
anomaly_rules.py, scene_rules.py, person_detector.py와 같이 두고 실행한다.

실행 방법
---------
  python3 worker_skeleton.py                          # CSI 카메라로 실시간
  python3 worker_skeleton.py --source test.mp4        # 테스트 영상 파일로 (권장)
  python3 worker_skeleton.py --source test.mp4 --loop # 영상 반복 재생
  python3 worker_skeleton.py --no-report              # 서버 전송 없이 판정만 (튜닝용)
  python3 worker_skeleton.py --after-hours            # 04 무단침입 테스트 (영업시간 외로 강제)
  python3 worker_skeleton.py --loiter 20              # 05 장시간체류 임계를 20초로 (테스트용)

디버그 화면
-----------
실행하면 8090 포트로 MJPEG 스트림이 열린다. 브라우저에서
  - 젯슨 VNC 안에서:  http://localhost:8090/
  - 같은 네트워크의 내 PC에서: http://10.100.0.164:8090/
로 접속하면 박스/ID/속도/판정 진행상황이 실시간으로 보인다.

※ cv2.imshow를 쓰지 않는 이유: 젯슨 OpenCV가 GTK2로 빌드돼 있어서 GTK3 라이브러리와 같은
   프로세스에 로드되면 "GTK+ 2.x symbols detected" 에러로 코어덤프가 난다. 그래서 창을 띄우는
   대신 표준 라이브러리(http.server)만으로 MJPEG 스트리밍 서버를 띄운다(추가 설치 없음).

판정 담당 (CCTV_ISSUE_CODE 기준)
--------------------------------
  01 폭행       : anomaly_rules.py  (사람 쌍 겹침 + 격한 움직임 점수 누적)
  02 기물파손   : scene_rules.py    (배경차분 - 사람과 무관한 큰 변화가 남아있는지)
  03 쓰러짐     : anomaly_rules.py  (박스 가로/세로 비율 + 지속시간)
  04 무단침입   : anomaly_rules.py  (영업시간 외 사람 탐지)
  05 장시간체류 : anomaly_rules.py  (한 트랙이 화면에 머문 시간)
  06 화재       : scene_rules.py    (불꽃색 비율 + 깜빡임)  ※ DB에 06 코드 등록돼 있어야 전송됨

Python 3.6 환경이라 `from __future__ import annotations`(3.7+)는 쓰지 않는다.
"""

import argparse
import queue
import socketserver
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Dict, List, Optional, Tuple

import cv2
import requests

from anomaly_rules import Box, TrackManager, AnomalyEvent, LOITER_SEC
from person_detector import get_person_boxes as _yolo_get_person_boxes
from scene_rules import DamageDetector, FireDetector
from homography import get_homography
from incident import IncidentTracker
from visitor import VisitorTracker

# ---------------------------------------------------------------------------
# 설정
# ---------------------------------------------------------------------------

# 젯슨 CSI 카메라(IMX219)는 V4L2로 직접 열면 초록 화면만 나온다(ISP 미경유).
# 반드시 nvarguscamerasrc GStreamer 파이프라인으로 열어야 정상 컬러가 나옴.
CSI_PIPELINE = (
    "nvarguscamerasrc ! "
    "video/x-raw(memory:NVMM), width=1280, height=720, framerate=30/1 ! "
    "nvvidconv flip-method=0 ! "
    "video/x-raw, width=1280, height=720, format=BGRx ! "
    "videoconvert ! "
    "video/x-raw, format=BGR ! appsink"
)

CNO = 1                      # 이 Jetson이 담당하는 CCTV 번호 (CCTV_STREAM 연동 시 교체)
BUSINESS_HOURS = (9, 22)     # 영업시간 (04 판정용)

FASTAPI_BASE_URL = "http://139.150.91.194:11200"
REPORT_ISSUE_ENDPOINT = FASTAPI_BASE_URL + "/api/cctv/issue/report"
VISITOR_ENTER_ENDPOINT = FASTAPI_BASE_URL + "/api/cctv/visitor/enter"
VISITOR_EXIT_ENDPOINT = FASTAPI_BASE_URL + "/api/cctv/visitor/exit"
REPORT_TIMEOUT_SEC = 60      # 서버가 LLM(comnet 생성)을 기다리므로 넉넉히. 비동기라 루프엔 영향 없음.

ENABLE_DAMAGE = True         # 02 기물파손 판정 on/off
ENABLE_FIRE = True           # 06 화재 판정 on/off (DB에 06 코드 없으면 서버가 400으로 거부함)


# ---------------------------------------------------------------------------
# 1) 사람 탐지
# ---------------------------------------------------------------------------

def get_person_boxes(frame) -> List[Tuple[Box, float]]:
    """프레임 한 장에서 사람 바운딩박스를 [((x1,y1,x2,y2), det_conf), ...]로 반환."""
    return _yolo_get_person_boxes(frame)


# ---------------------------------------------------------------------------
# 2) 추적 - IOU + 속도예측 + 짧은 미검출 허용
# ---------------------------------------------------------------------------

class SimpleTracker:
    """프레임 간 사람 박스를 이어붙여 같은 사람에게 같은 track_id를 주는 트래커.

    단순 IOU 매칭만 쓰면 젯슨의 낮은 FPS에서 사람이 조금만 빨리 움직여도 박스가 안 겹쳐서
    매 프레임 새 ID가 생긴다(=ID churn). 그러면 "같은 사람이 N초 유지" 같은 판정이 전부 깨진다.
    그래서 두 가지를 추가했다:

      1) 속도 예측: 직전 속도로 "이번 프레임엔 대략 여기 있겠다"를 계산해서, IOU가 안 겹쳐도
         예측 위치 근처(max_center_dist 이내)면 같은 사람으로 인정.
      2) 짧은 미검출 허용(max_lost_sec): YOLO가 한두 프레임 사람을 놓쳐도 트랙을 바로 버리지
         않고 잠깐 살려둔다. 다시 나타나면 같은 ID로 이어짐.

    ByteTrack만큼 정교하진 않지만 추가 설치 없이 위 두 문제를 직접 겨냥해서 완화한다.
    """

    def __init__(self, iou_match_thresh: float = 0.2,
                 max_center_dist: float = 180.0,
                 dist_box_factor: float = 2.0,
                 max_lost_sec: float = 1.5):
        self.next_id = 1
        self.tracks: Dict[int, dict] = {}   # tid -> {box, center, velocity(px/s), t}
        self.iou_match_thresh = iou_match_thresh
        self.max_center_dist = max_center_dist
        # 허용 거리를 박스 크기에 비례해서도 잡는다. 고정 픽셀값만 쓰면, 카메라에 가까워서 박스가
        # 큰 사람(=화면상 이동 거리도 큼)이 매 프레임 새 ID를 받게 된다. "박스 폭의 2배 이내면
        # 같은 사람"처럼 상대 기준을 같이 두면 카메라 거리와 무관하게 동작한다.
        self.dist_box_factor = dist_box_factor
        self.max_lost_sec = max_lost_sec

    @staticmethod
    def _iou(a: Box, b: Box) -> float:
        from anomaly_rules import iou as _iou_fn
        return _iou_fn(a, b)

    @staticmethod
    def _center(b: Box) -> Tuple[float, float]:
        x1, y1, x2, y2 = b
        return (x1 + x2) / 2.0, (y1 + y2) / 2.0

    def update(self, boxes: List[Box], now: Optional[float] = None) -> List[Tuple[int, Box]]:
        now = now if now is not None else time.time()

        # [중요] 매칭은 "직전 프레임 상태(prev_tracks)"만 보고 하고, 결과는 별도 dict(updated)에 모은다.
        # 이번 프레임에 새로 만든 트랙을 self.tracks에 바로 넣으면서 매칭 대상으로도 쓰면,
        #   (1) 예측 위치(predicted)에 없는 ID를 조회해서 KeyError가 나고
        #   (2) 같은 프레임의 두 사람이 같은 트랙에 매칭될 수도 있다(있을 수 없는 일).
        # "읽는 대상"과 "쓰는 대상"을 분리하는 건 이런 종류의 버그를 막는 기본 패턴이다.
        prev_tracks = self.tracks

        # 각 기존 트랙의 "이번 프레임 예상 위치"를 직전 속도로 미리 계산
        predicted = {}
        for tid, info in prev_tracks.items():
            dt = max(0.0, now - info["t"])
            vx, vy = info["velocity"]
            cx, cy = info["center"]
            predicted[tid] = (cx + vx * dt, cy + vy * dt)

        matched_ids = set()
        updated: Dict[int, dict] = {}
        results: List[Tuple[int, Box]] = []

        for box in boxes:
            cx, cy = self._center(box)
            box_w = max(1.0, box[2] - box[0])
            allow_dist = max(self.max_center_dist, box_w * self.dist_box_factor)
            best_id, best_score = None, None

            for tid, info in prev_tracks.items():
                if tid in matched_ids:
                    continue
                iou_score = self._iou(box, info["box"])
                pcx, pcy = predicted[tid]
                dist = ((cx - pcx) ** 2 + (cy - pcy) ** 2) ** 0.5

                if not (iou_score >= self.iou_match_thresh or dist <= allow_dist):
                    continue

                score = iou_score - (dist / 10000.0)   # IOU 높고 예측 위치에 가까울수록 우선
                if best_score is None or score > best_score:
                    best_id, best_score = tid, score

            if best_id is not None:
                tid = best_id
                matched_ids.add(tid)
                prev = prev_tracks[tid]
                dt = max(1e-3, now - prev["t"])
                vx = (cx - prev["center"][0]) / dt
                vy = (cy - prev["center"][1]) / dt
            else:
                tid = self.next_id
                self.next_id += 1
                vx, vy = 0.0, 0.0

            updated[tid] = {"box": box, "center": (cx, cy), "velocity": (vx, vy), "t": now}
            results.append((tid, box))

        # 이번 프레임에 매칭 안 된 직전 트랙: max_lost_sec 안이면 살려둔다
        # (YOLO가 한두 프레임 놓쳐도 같은 ID로 이어지게 하려는 것)
        for tid, info in prev_tracks.items():
            if tid in updated:
                continue
            if now - info["t"] <= self.max_lost_sec:
                updated[tid] = info

        self.tracks = updated
        return results


# ---------------------------------------------------------------------------
# 3) 이벤트 전송 - 백그라운드 스레드 + 큐 (메인 루프를 절대 막지 않음)
# ---------------------------------------------------------------------------
#
# [왜 이렇게 바꿨나]
#   기존에는 report_issue() 안에서 requests.post()를 바로 호출했다. 그런데 서버는 이벤트를
#   받으면 comnet 문장을 만들려고 LLM(Ollama gemma)을 호출하는데 이게 10초 이상 걸린다.
#   즉 이벤트 1건 = 메인 루프 10초 정지 = 카메라 화면 멈춤 + 그 사이 프레임 전부 유실.
#   (실무에서 흔한 패턴: "느린 I/O를 이벤트 루프 안에서 동기 호출" -> 전체가 그 속도에 묶임)
#
#   해결: 생산자-소비자 패턴. 메인 루프는 큐에 넣기만 하고(마이크로초) 즉시 다음 프레임으로 가고,
#   별도 스레드가 큐에서 꺼내 느긋하게 HTTP 전송한다. 파이썬 GIL이 있어도 requests가 네트워크
#   응답을 기다리는 동안엔 GIL을 놓기 때문에 메인 루프의 추론은 정상적으로 계속 돈다.
#   (Java의 ExecutorService에 작업 submit하는 것과 같은 구조. PHP는 이런 게 어려워서 보통
#    큐 테이블이나 Redis에 넣고 크론/워커가 처리하는데, 개념은 똑같다.)

_event_queue = queue.Queue(maxsize=100)
_enable_report = True
FRAME_SIZE = [None, None]      # 카메라 해상도 (w, h) - main()에서 채운다


def _post_visitor(kind: str, data: dict) -> None:
    """손님 입장/퇴장을 서버에 보낸다. 이슈와 같은 큐를 쓰므로 메인 루프는 막히지 않는다."""
    url = VISITOR_ENTER_ENDPOINT if kind == "enter" else VISITOR_EXIT_ENDPOINT
    try:
        res = requests.post(url, json=data, timeout=REPORT_TIMEOUT_SEC)
        res.raise_for_status()
        if kind == "enter":
            print("[visitor] 입장 %s -> %s" % (data["trackId"], res.status_code))
        else:
            print("[visitor] 퇴장 %s (%d분, state=%d) -> %s"
                  % (data["trackId"], data["staytime"], data["state"], res.status_code))
    except requests.RequestException as e:
        print("[visitor] %s 전송 실패: %s" % (kind, e))


def _post_event(event: AnomalyEvent) -> None:
    payload = {
        "cno": CNO,
        "code": event.code,
        "detail": event.detail,          # 서버가 comnet으로 다듬는 원문
        "confidence": event.confidence,  # 서버가 reliability로 그대로 저장
        "trackIds": event.track_ids,
        "detectedAt": event.at,
        # 상황 단위 중복 억제 정보 (incident.py). 서버/알림 쪽에서 최초 발생과
        # 장기 미조치 재알림을 구분할 수 있게 함께 보낸다.
        "stage": getattr(event, "stage", "new"),
        "duration": round(getattr(event, "duration", 0.0), 1),
    }

    # 이슈 발생 위치.
    #
    # [역할 분담]
    #   imgX/imgY/imgW/imgH : "카메라 화면의 어디서 일어났나" - 젯슨만 알 수 있는 정보라 항상 보낸다.
    #                         설정이 전혀 필요 없고, 받는 쪽이 이 값으로 직접 도면 변환을 할 수 있다.
    #   x/y                 : "도면의 어디인가" - homography.json이 있을 때만 보내는 부가 정보(0~1 비율).
    #                         도면이 바뀌면 변환도 바뀌므로, 원칙적으로는 도면을 가진 서버 쪽에서
    #                         변환하는 게 유지보수에 유리하다. 여기서 보내는 건 편의 제공일 뿐이다.
    #
    # imgW/imgH를 같이 보내는 이유: 받는 쪽이 좌표를 해석하려면 기준 해상도를 알아야 한다.
    # 카메라 설정이 바뀌어 해상도가 달라져도 값의 의미가 깨지지 않게 하는 안전장치.
    if event.point is not None:
        payload["imgX"], payload["imgY"] = round(event.point[0], 1), round(event.point[1], 1)
        if FRAME_SIZE[0]:
            payload["imgW"], payload["imgH"] = FRAME_SIZE[0], FRAME_SIZE[1]
        plan = get_homography().to_plan(event.point)
        if plan is not None:
            payload["x"], payload["y"] = plan[0], plan[1]
    try:
        res = requests.post(REPORT_ISSUE_ENDPOINT, json=payload, timeout=REPORT_TIMEOUT_SEC)
        res.raise_for_status()
        pos = ""
        if "x" in payload:
            pos = " 도면좌표=(%.1f, %.1f)" % (payload["x"], payload["y"])
        print("[report] code=%s conf=%s%s -> %s" % (event.code, event.confidence, pos, res.status_code))
    except requests.RequestException as e:
        # 안전 이벤트라 전송 실패해도 워커는 계속 돌아야 함 (재시도 큐는 추후 보강)
        print("[report] 전송 실패: %s (code=%s)" % (e, event.code))


def _sender_loop() -> None:
    while True:
        item = _event_queue.get()
        if item is None:       # 종료 신호
            break
        try:
            kind, data = item
            if kind == "issue":
                _post_event(data)
            else:
                _post_visitor(kind, data)
        except Exception as e:
            print("[report] 전송 스레드 예외: %s" % e)
        finally:
            _event_queue.task_done()


def start_sender_thread() -> None:
    t = threading.Thread(target=_sender_loop)
    t.daemon = True
    t.start()


def _enqueue(kind: str, data) -> None:
    if not _enable_report:
        return
    try:
        _event_queue.put_nowait((kind, data))
    except queue.Full:
        print("[report] 전송 큐가 가득 참 - %s 버림 (서버 응답이 너무 느린 상태)" % kind)


def report_issue(event: AnomalyEvent) -> None:
    """큐에 넣기만 하고 즉시 리턴 (메인 루프 정지 없음)."""
    stage = getattr(event, "stage", "new")
    tag = "재알림" if stage == "escalation" else "신규"
    print("[event][%s] code=%s conf=%.1f | %s" % (tag, event.code, event.confidence, event.detail))
    _enqueue("issue", event)


def report_visitor(kind: str, data: dict) -> None:
    """손님 입장/퇴장을 큐에 넣는다."""
    _enqueue(kind, data)


# ---------------------------------------------------------------------------
# 4) 디버그 화면 스트리밍 (표준 라이브러리만 사용)
# ---------------------------------------------------------------------------

_latest_jpeg_lock = threading.Lock()
_latest_jpeg = None


def _update_debug_frame(frame) -> None:
    global _latest_jpeg
    ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
    if ok:
        with _latest_jpeg_lock:
            _latest_jpeg = buf.tobytes()


_INDEX_HTML = b"""<html><head><title>CCTV worker debug</title>
<style>body{margin:0;background:#111;color:#eee;font-family:sans-serif}
img{width:100%;max-width:1280px;display:block;margin:0 auto}</style></head>
<body><img src="/stream"></body></html>"""


class _MJPEGHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path != "/stream":
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(_INDEX_HTML)))
            self.end_headers()
            self.wfile.write(_INDEX_HTML)
            return

        self.send_response(200)
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        self.end_headers()
        try:
            while True:
                with _latest_jpeg_lock:
                    jpg = _latest_jpeg
                if jpg is not None:
                    self.wfile.write(b"--frame\r\n")
                    self.wfile.write(b"Content-Type: image/jpeg\r\n")
                    self.wfile.write(("Content-Length: %d\r\n\r\n" % len(jpg)).encode())
                    self.wfile.write(jpg)
                    self.wfile.write(b"\r\n")
                time.sleep(0.05)
        except Exception:
            pass   # 브라우저 탭 닫으면 여기로 옴 (정상)

    def log_message(self, fmt, *args):
        pass


class _ThreadingHTTPServer(socketserver.ThreadingMixIn, HTTPServer):
    daemon_threads = True
    allow_reuse_address = True


def start_debug_server(port: int) -> None:
    server = _ThreadingHTTPServer(("0.0.0.0", port), _MJPEGHandler)
    t = threading.Thread(target=server.serve_forever)
    t.daemon = True
    t.start()
    print("[worker] 디버그 화면: http://localhost:%d/  (다른 PC에서는 젯슨 IP:%d)" % (port, port))


def draw_debug_overlay(frame, tracked, manager, now, stats):
    """박스/ID/판정 진행상황을 그려서 눈으로 튜닝할 수 있게 한다. 판정 로직엔 영향 없음."""
    from anomaly_rules import (
        iou as _iou_fn, FALL_SUSTAIN_SEC, ASSAULT_IOU_THRESH,
        ASSAULT_MOTION_THRESH, ASSAULT_SCORE_TRIGGER,
    )

    # 1) 사람별 박스 + ID + 속도 + 가로세로비 + 체류시간 + 쓰러짐 진행상황
    for tid, box in tracked:
        x1, y1, x2, y2 = [int(v) for v in box]
        track = manager.tracks.get(tid)
        motion = track.motion_intensity(now) if track else 0.0
        w, h = max(1, x2 - x1), max(1, y2 - y1)
        aspect = float(w) / h
        stay = track.duration(now) if track else 0.0

        fall_txt = ""
        if track and track.fall_state_since is not None:
            fall_txt = " FALL %.1f/%.1fs" % (now - track.fall_state_since, FALL_SUSTAIN_SEC)

        color = (0, 0, 255) if fall_txt else (0, 255, 0)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, "ID%d mot=%.1f ar=%.2f stay=%.0fs%s" % (tid, motion, aspect, stay, fall_txt),
                    (x1, max(15, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # 2) 겹친 사람 쌍은 선으로 연결 + IOU/속도 표시 (폭행 판정 근거를 눈으로 보기 위함)
    ids = [tid for tid, _ in tracked]
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            ta, tb = manager.tracks.get(ids[i]), manager.tracks.get(ids[j])
            if not ta or not tb:
                continue
            ba, bb = ta.latest_box(), tb.latest_box()
            if ba is None or bb is None:
                continue
            overlap = _iou_fn(ba, bb)
            if overlap <= 0:
                continue
            motion = max(ta.motion_intensity(now), tb.motion_intensity(now))
            ca = (int((ba[0] + ba[2]) / 2), int((ba[1] + ba[3]) / 2))
            cb = (int((bb[0] + bb[2]) / 2), int((bb[1] + bb[3]) / 2))
            hot = overlap >= ASSAULT_IOU_THRESH and motion >= ASSAULT_MOTION_THRESH
            col = (0, 0, 255) if hot else (0, 200, 200)
            cv2.line(frame, ca, cb, col, 2)
            cv2.putText(frame, "iou=%.2f mot=%.1f" % (overlap, motion),
                        ((ca[0] + cb[0]) // 2, (ca[1] + cb[1]) // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 2)

    # 3) 상단: 사람 수 / FPS / 폭행 점수 게이지
    cv2.putText(frame, "people=%d  visitors=%d  fps=%.1f  queue=%d" % (
        len(tracked), stats.get("visitors", 0), stats.get("fps", 0.0), _event_queue.qsize()),
        (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    incident_txt = stats.get("incidents", "")
    if incident_txt:
        cv2.putText(frame, incident_txt, (10, frame.shape[0] - 38),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 165, 255), 2)

    score = manager.assault_score
    ratio = min(1.0, score / ASSAULT_SCORE_TRIGGER)
    cv2.rectangle(frame, (10, 35), (210, 50), (80, 80, 80), 1)
    cv2.rectangle(frame, (10, 35), (10 + int(200 * ratio), 50),
                  (0, 0, 255) if ratio >= 1.0 else (0, 165, 255), -1)
    cv2.putText(frame, "assault %.1f/%.1f %s" % (score, ASSAULT_SCORE_TRIGGER,
                                                 manager.assault_last_info),
                (220, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 2)

    # 4) 하단: 기물파손/화재 지표
    cv2.putText(frame, "damage=%.1f%%  fire=%.1f%% flicker=%.2f" % (
        stats.get("damage_ratio", 0.0) * 100,
        stats.get("fire_ratio", 0.0) * 100,
        stats.get("fire_flicker", 0.0)),
        (10, frame.shape[0] - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 2)
    return frame


# ---------------------------------------------------------------------------
# 5) 입력 소스 열기
# ---------------------------------------------------------------------------

def open_capture(source: str):
    if source == "camera":
        cap = cv2.VideoCapture(CSI_PIPELINE, cv2.CAP_GSTREAMER)
    else:
        cap = cv2.VideoCapture(source)     # 영상 파일 / RTSP 주소
    if not cap.isOpened():
        raise RuntimeError("입력을 열 수 없습니다: %s" % source)
    return cap


# ---------------------------------------------------------------------------
# 메인 루프
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="젯슨 CCTV 이상행동 감지 워커")
    p.add_argument("--source", default="camera",
                   help="camera(기본, CSI 카메라) 또는 테스트 영상 파일 경로/RTSP 주소")
    p.add_argument("--loop", action="store_true", help="영상 파일이 끝나면 처음부터 반복")
    p.add_argument("--no-report", action="store_true", help="서버 전송 없이 판정 로그만 출력")
    p.add_argument("--after-hours", action="store_true",
                   help="현재 시각과 무관하게 영업시간 외로 간주 (04 무단침입 테스트용)")
    p.add_argument("--loiter", type=float, default=LOITER_SEC,
                   help="05 장시간체류 임계(초). 테스트할 땐 20 정도로 낮춰서 확인")
    p.add_argument("--no-display", action="store_true",
                   help="디버그 화면을 띄우지 않는다(그냥 돌리기용). 오버레이 연산이 빠져서 조금 더 빠름")
    p.add_argument("--port", type=int, default=8090, help="디버그 화면 포트")
    return p.parse_args()


def main() -> None:
    global _enable_report
    args = parse_args()
    _enable_report = not args.no_report

    is_video_file = args.source != "camera"
    cap = open_capture(args.source)
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or None
    FRAME_SIZE[0] = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or None
    FRAME_SIZE[1] = frame_height

    tracker = SimpleTracker()
    manager = TrackManager(frame_height=frame_height, loiter_sec=args.loiter)
    damage_detector = DamageDetector()
    fire_detector = FireDetector()
    incidents = IncidentTracker()              # 같은 상황 반복 전송 억제
    visitors = VisitorTracker(cno=CNO, long_stay_sec=args.loiter)

    show_display = not args.no_display
    if show_display:
        start_debug_server(args.port)
    else:
        print("[worker] 디버그 화면 없이 실행 (--no-display)")
    start_sender_thread()
    get_homography()     # homography.json 로드 (없으면 경고만 출력하고 좌표 없이 동작)

    print("[worker] 시작 (source=%s, report=%s, loiter=%.0fs) - Ctrl+C로 종료"
          % (args.source, _enable_report, args.loiter))

    wall_start = time.time()
    last_status_at = 0.0
    fps, frame_count, fps_t0 = 0.0, 0, time.time()
    stats = {"fps": 0.0, "damage_ratio": 0.0, "fire_ratio": 0.0, "fire_flicker": 0.0}

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                if is_video_file and args.loop:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)   # 처음부터 다시
                    continue
                if is_video_file:
                    print("[worker] 영상 끝")
                    break
                print("[worker] 프레임 읽기 실패, 재시도")
                time.sleep(0.5)
                continue

            # 시간 기준: 카메라는 실제 시각, 영상 파일은 영상 내부 타임스탬프를 쓴다.
            # (영상은 추론 속도 때문에 실시간보다 느리게 재생되는데, 실제 시각으로 재면
            #  "3초간 쓰러진 자세 유지" 같은 판정이 영상 내용과 어긋나기 때문)
            if is_video_file:
                pos_ms = cap.get(cv2.CAP_PROP_POS_MSEC) or 0.0
                now = wall_start + pos_ms / 1000.0
            else:
                now = time.time()

            detections = get_person_boxes(frame)
            boxes_only = [box for box, _ in detections]
            tracked = tracker.update(boxes_only, now=now)

            manager.update(tracked, now=now)
            events = manager.check_all(now=now,
                                       business_hours=BUSINESS_HOURS,
                                       force_after_hours=args.after_hours)

            # 02 기물파손 / 06 화재는 사람 추적과 무관하게 프레임 전체를 보고 판정
            if ENABLE_DAMAGE:
                dmg_event, dmg_ratio = damage_detector.update(frame, boxes_only, now)
                stats["damage_ratio"] = dmg_ratio
                if dmg_event:
                    events.append(dmg_event)
            if ENABLE_FIRE:
                fire_event, fire_ratio, flicker = fire_detector.update(frame, now)
                stats["fire_ratio"], stats["fire_flicker"] = fire_ratio, flicker
                if fire_event:
                    events.append(fire_event)

            # 손님 입·퇴장 집계 (CCTV_VISITOR)
            enters, exits = visitors.update(tracked, now=now)
            for e in enters:
                report_visitor("enter", e)
            for e in exits:
                report_visitor("exit", e)

            # [중요] 룰 엔진이 낸 이벤트를 그대로 보내지 않고 상황 단위로 한 번 거른다.
            # 쓰러진 사람이 계속 누워 있으면 룰은 매번 감지하지만, 여기서 최초 1회와
            # 정해진 재알림 시점만 통과시킨다. (incident.py 참고)
            for ev in incidents.process(events, now=now):
                report_issue(ev)

            # FPS 계산 (실제 처리 속도. 낮으면 추적이 불안정해지므로 튜닝 지표로 중요)
            frame_count += 1
            if frame_count >= 10:
                elapsed = time.time() - fps_t0
                fps = frame_count / elapsed if elapsed > 0 else 0.0
                stats["fps"] = fps
                frame_count, fps_t0 = 0, time.time()

            if show_display:
                stats["visitors"] = visitors.active_count
                stats["incidents"] = incidents.status_line()
                debug_frame = draw_debug_overlay(frame.copy(), tracked, manager, now, stats)
                _update_debug_frame(debug_frame)

            if time.time() - last_status_at >= 5.0:
                extra = incidents.status_line()
                print("[worker] 사람 %d명(손님 %d) | fps %.1f | 폭행점수 %.1f | 전송대기 %d %s"
                      % (len(tracked), visitors.active_count, fps,
                         manager.assault_score, _event_queue.qsize(), extra))
                last_status_at = time.time()

    except KeyboardInterrupt:
        print("\n[worker] 종료")
    finally:
        # 아직 '입장중'으로 남아있는 손님들을 퇴장 처리한다.
        # 안 하면 STATE=0 행이 DB에 영원히 남아서 "현재 매장 인원"이 계속 늘어난 것처럼 보인다.
        try:
            for e in visitors.flush():
                report_visitor("exit", e)
            _event_queue.join()        # 큐에 남은 전송이 끝날 때까지 잠깐 대기
        except Exception as e:
            print("[worker] 종료 처리 중 오류: %s" % e)
        cap.release()


if __name__ == "__main__":
    main()
