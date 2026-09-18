# -*- coding: utf-8 -*-
"""
anomaly_rules.py

CCTV_ISSUE_CODE 01(폭행) / 03(쓰러짐·응급) / 04(무단침입) / 05(장시간체류) 판정용 휴리스틱 룰 엔진.
학습 데이터/모델 없이, YOLO+트래커가 뽑아준 사람 바운딩박스만으로 판단한다.

- 02(기물파손) / 06(화재)는 사람 박스로 판단할 수 없어서 scene_rules.py(프레임 전체 기반)가 담당.
- 05(장시간체류)는 서버(입장~퇴장 시각 계산)도 따로 계산하지만, 젯슨에서도 "한 사람이 화면 안에
  N분 이상 머무름"으로 독립 판정한다(데모/테스트에서 서버 방문 데이터 없이도 확인 가능하도록).

[2026-09-14 주요 변경 - 폭행(01) 판정 방식을 근본적으로 바꿈]
  기존: "같은 track_id 쌍이 N초 연속으로 (겹침 + 빠른 움직임)" 조건을 만족해야 확정.
  문제: 젯슨의 낮은 FPS에서 사람이 빠르게 움직이면(=폭행 상황 그 자체) track_id가 계속 바뀌어서
        "같은 쌍이 연속 N초"라는 조건이 물리적으로 성립하지 못함 -> 미탐.
        반대로 임계값을 낮추면 ID가 바뀔 때마다 쿨다운이 초기화돼서 스팸 발생.
  변경: track_id에 의존하지 않는 **카메라 단위 누적 점수(assault_score)** 방식으로 전환.
        - 매 프레임, "겹쳐 있으면서 빠르게 움직이는 사람 쌍"이 하나라도 있으면 점수 +1
        - 시간이 지나면 점수는 초당 일정량씩 감소(decay)
        - 점수가 ASSAULT_SCORE_TRIGGER를 넘으면 확정
        즉 "짧은 시간 안에 격한 접촉 프레임이 여러 번 누적되면 폭행"으로 보는 것이라,
        중간에 ID가 바뀌거나 한두 프레임 탐지를 놓쳐도 증거가 사라지지 않는다.
        (원리는 스팸 필터의 점수 누적과 같음: 한 방에 판단하지 않고 증거를 모아서 임계값 초과 시 확정)

주의: 아래 임계값들은 전부 "시작점"이다. 카메라 각도/해상도/거리에 맞춰 실측 튜닝 필요.
Python 3.6 환경(Jetson Nano)이라 dataclasses 백포트 필요: pip3 install dataclasses
"""

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional, Tuple

Box = Tuple[float, float, float, float]  # x1, y1, x2, y2 (픽셀 좌표)

# ---------------------------------------------------------------------------
# 임계값 (전부 튜닝 대상 - 디버그 화면에 찍히는 숫자 보면서 조정할 것)
# ---------------------------------------------------------------------------

# --- 03 쓰러짐 ---
FALL_ASPECT_RATIO = 1.3          # 박스 가로/세로 비율이 이 값을 넘으면 "누운 자세"
FALL_SUSTAIN_SEC = 3.0           # 그 자세가 이만큼(초) 유지돼야 확정 (숙이는 동작과 구분)
FALL_MAX_HEIGHT_RATIO = 0.35     # 화면 대비 박스 높이가 이 비율 이하여야 "낮게 깔림"

# --- 01 폭행 (점수 누적 방식) ---
ASSAULT_IOU_THRESH = 0.10        # 두 사람 박스가 이 이상 겹치면 "붙어있다"
# [2026-09-14 변경] "중심 이동속도(px/s)" -> "몸 크기 대비 움직임 세기(키배수/초)"로 교체.
#   실제 폭행 영상을 분석해보니 가해자는 제자리에 서서 팔만 휘두르기 때문에 박스 '중심'은
#   거의 안 움직인다(그래서 폭행점수가 계속 0이었음). 대신 팔이 뻗어질 때마다 박스 '폭'이
#   출렁이므로, 중심 이동 + 박스 형태 변화(|Δw|+|Δh|)를 합쳐서 보고, 사람 키로 나눠 정규화한다.
#   정규화하는 이유: 카메라에서 멀면 같은 동작도 픽셀 수가 작아진다. 키로 나누면 거리와
#   무관하게 "몸 크기의 몇 배만큼 움직였나"라는 동일한 기준이 된다.
#   실측(위 영상): 폭행 구간 중앙값 약 2.6 키배수/초. 일반 보행은 0.8 안팎.
ASSAULT_MOTION_THRESH = 1.2      # 키배수/초. 이 이상이면 "격한 움직임"

# [2026-09-18 변경] "붙어있다" 판정을 IOU 단독 -> IOU 또는 간격비율로 확장.
#   실제 싸움 영상(편의점 내부, 하향 각도 카메라) 실측 결과:
#     IOU        중앙값 0.000 / 최대 0.128  -> IOU>=0.10 만족: 67프레임 중 3프레임 (4%)
#     간격/키    중앙값 0.067                -> <=0.5 만족: 67프레임 중 59프레임 (88%)
#   즉 마주보고 싸우는 두 사람은 박스가 '거의 안 겹친다'. 카메라를 위에서 비스듬히 내려보면
#   두 사람이 좌우로 나란히 서기 때문에 박스가 옆으로 붙기만 하고 겹치지 않는 것이다.
#   그래서 IOU 하나로 근접을 판정하면 원리적으로 폭행을 잡을 수 없다.
#   대신 "두 박스 사이의 빈 간격"을 사람 키로 나눈 값을 쓴다. 키로 나누는 이유는 움직임
#   세기와 동일하다 - 카메라에서 멀면 같은 거리도 픽셀 수가 작아지므로, 키 대비로 보면
#   거리와 무관하게 "팔 뻗으면 닿는 거리인가"를 판정할 수 있다(사람 팔 길이 ≈ 키의 0.4배).
ASSAULT_GAP_RATIO = 0.5          # 박스 사이 빈 간격 / 평균 키. 이 이하면 "팔 닿는 거리"
ASSAULT_CDIST_RATIO = 1.3        # 박스 중심거리 / 평균 키. 간격이 0이어도 중심이 너무 멀면 제외
# 단안(单眼) 카메라에서 깊이를 추정하는 가장 싼 방법은 "겉보기 크기"다. 같은 사람이라도
# 카메라에 가까우면 박스가 크고 멀면 작다. 즉 두 박스의 키 비율이 크게 다르면 두 사람은
# 서로 다른 거리에 있다는 뜻이므로, 화면상 겹쳐 보여도 실제로는 닿을 수 없다.
# (카메라 앞을 지나가는 사람 + 저 뒤 진열대 앞의 사람 → 화면에선 겹침, 실제로는 무관)
ASSAULT_HEIGHT_RATIO_MAX = 2.0   # 큰 박스 키 / 작은 박스 키. 이 이상 차이나면 다른 거리로 판단

# [2026-09-18 변경] 점수 누적을 "프레임당 +1" -> "시간당 +N"으로 교체.
#   프레임당 점수는 FPS에 따라 임계 도달 시간이 달라진다(8fps면 0.75초, 4fps면 1.5초).
#   시간 기반이면 "격한 접촉이 누적 N초"라는 물리적 의미가 고정된다.
ASSAULT_SCORE_PER_SEC = 1.0      # 조건 만족 중에는 초당 이만큼 누적
ASSAULT_SCORE_DECAY_PER_SEC = 0.8  # 조건이 끊기면 초당 이만큼씩 감소
ASSAULT_SCORE_TRIGGER = 2.0      # 누적 2.0 = "격한 접촉이 합계 2초" -> 폭행 확정
ASSAULT_SCORE_MAX = 4.0          # 점수 상한 (너무 쌓여서 한참 안 내려가는 것 방지)

# --- 05 장시간체류 ---
LOITER_SEC = 300.0               # 한 사람이 화면에 이만큼(초) 머물면 체류 (기본 5분)
                                 # 데모/테스트할 땐 20~30초로 낮춰서 확인할 것

# --- 공통 쿨다운 ---
EVENT_COOLDOWN_SEC = 60.0        # 같은 트랙 + 같은 코드 재전송 방지
GLOBAL_EVENT_COOLDOWN_SEC = 30.0 # 카메라 단위 최종 방어선 (track ID가 바뀌어도 스팸 차단)

# 트랙 히스토리에서 속도 계산에 사용할 시간창
SPEED_WINDOW_SEC = 1.0


def iou(a: Box, b: Box) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def center(b: Box) -> Tuple[float, float]:
    x1, y1, x2, y2 = b
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def mean_height(a: Box, b: Box) -> float:
    """두 사람의 평균 키(픽셀). 거리 정규화의 분모로 쓴다."""
    return max(1.0, ((a[3] - a[1]) + (b[3] - b[1])) / 2.0)


def gap_ratio(a: Box, b: Box) -> float:
    """두 박스 사이의 '빈 간격'을 평균 키로 나눈 값.

    겹쳐 있으면 0. 옆으로 나란히 붙어만 있어도 0에 가깝다.
    IOU와 달리 '겹침'을 요구하지 않으므로, 마주보고 싸우는 두 사람처럼
    박스가 옆으로 닿기만 하는 상황도 근접으로 잡힌다.
    """
    dx = max(0.0, max(a[0], b[0]) - min(a[2], b[2]))
    dy = max(0.0, max(a[1], b[1]) - min(a[3], b[3]))
    gap = (dx * dx + dy * dy) ** 0.5
    return gap / mean_height(a, b)


def center_dist_ratio(a: Box, b: Box) -> float:
    """두 박스 중심 사이 거리를 평균 키로 나눈 값.

    gap_ratio 단독으로는 '한 사람이 다른 사람 박스를 통째로 품는' 경우
    (예: 카메라 앞을 지나가는 사람과 멀리 있는 사람이 화면상 겹침)도
    0이 나온다. 중심거리로 한 번 더 걸러서 그런 경우를 배제한다.
    """
    ca, cb = center(a), center(b)
    d = ((ca[0] - cb[0]) ** 2 + (ca[1] - cb[1]) ** 2) ** 0.5
    return d / mean_height(a, b)


@dataclass
class Track:
    track_id: int
    history: Deque[Tuple[float, Box]] = field(default_factory=lambda: deque(maxlen=90))
    first_seen: float = 0.0
    last_seen: float = 0.0

    # 상태가 처음 감지된 시각(지속시간 판정용). 조건이 깨지면 None으로 리셋.
    fall_state_since: Optional[float] = None

    # 코드별 마지막 이벤트 전송 시각 (쿨다운용)
    last_event_at: Dict[str, float] = field(default_factory=dict)

    def push(self, box: Box, now: float) -> None:
        self.history.append((now, box))
        self.last_seen = now
        if self.first_seen == 0.0:
            self.first_seen = now

    def latest_box(self) -> Optional[Box]:
        return self.history[-1][1] if self.history else None

    def duration(self, now: float) -> float:
        """이 사람이 화면에 머문 시간(초)."""
        return now - self.first_seen if self.first_seen else 0.0

    def speed_px_per_sec(self, now: float) -> float:
        """최근 SPEED_WINDOW_SEC 동안의 중심점 이동 속도(px/sec). 디버그 표시/참고용."""
        pts = [(t, center(b)) for t, b in self.history if now - t <= SPEED_WINDOW_SEC]
        if len(pts) < 2:
            return 0.0
        (t0, c0), (t1, c1) = pts[0], pts[-1]
        dt = t1 - t0
        if dt <= 0:
            return 0.0
        dist = ((c1[0] - c0[0]) ** 2 + (c1[1] - c0[1]) ** 2) ** 0.5
        return dist / dt

    def motion_intensity(self, now: float) -> float:
        """최근 구간의 '몸 크기 대비' 움직임 세기 (단위: 키배수/초).

        중심 이동만 보면 제자리에서 팔을 휘두르는 동작(=폭행의 실제 모습)을 놓친다.
        그래서 프레임 사이의 (중심 이동거리 + 박스 폭 변화 + 박스 높이 변화)를 전부 더하고,
        사람 키로 나눠서 카메라 거리와 무관한 값으로 만든다.

        - 가만히 서 있음      : 약 0.0 ~ 0.2
        - 평범하게 걸어감      : 약 0.5 ~ 0.9
        - 때리기/몸싸움/넘어짐 : 약 1.5 이상 (실측 영상 중앙값 2.6)
        """
        pts = [(t, b) for t, b in self.history if now - t <= SPEED_WINDOW_SEC]
        if len(pts) < 2:
            return 0.0

        total, heights = 0.0, []
        for (t0, b0), (t1, b1) in zip(pts, pts[1:]):
            c0, c1 = center(b0), center(b1)
            moved = ((c1[0] - c0[0]) ** 2 + (c1[1] - c0[1]) ** 2) ** 0.5
            dw = abs((b1[2] - b1[0]) - (b0[2] - b0[0]))   # 팔을 뻗으면 폭이 늘어남
            dh = abs((b1[3] - b1[1]) - (b0[3] - b0[1]))   # 숙이거나 쓰러지면 높이가 변함
            total += moved + dw + dh
            heights.append(max(1.0, b1[3] - b1[1]))

        dt = pts[-1][0] - pts[0][0]
        if dt <= 0:
            return 0.0
        avg_height = sum(heights) / len(heights)
        return (total / dt) / avg_height

    def can_emit(self, code: str, now: float) -> bool:
        last = self.last_event_at.get(code)
        return last is None or (now - last) >= EVENT_COOLDOWN_SEC

    def mark_emitted(self, code: str, now: float) -> None:
        self.last_event_at[code] = now


def ground_point(b: Box) -> Tuple[float, float]:
    """박스에서 '바닥에 닿는 점'(하단 중앙 = 발 위치).

    호모그래피로 도면 좌표를 구할 때 반드시 이 점을 써야 한다. 박스 중심은 몸통 높이라
    공중에 떠 있는 점이어서, 도면에 옮기면 카메라에서 먼 쪽으로 밀려난 위치가 나온다.
    """
    x1, y1, x2, y2 = b
    return ((x1 + x2) / 2.0, y2)


@dataclass
class AnomalyEvent:
    code: str          # '01' | '02' | '03' | '04' | '05' | '06'
    track_ids: List[int]
    confidence: float  # 0~100
    detail: str        # Jetson이 판단한 근거 (서버가 comnet으로 다듬는 원문)
    at: float
    # 이벤트가 발생한 '카메라 화면상' 위치 (x, y). 워커가 호모그래피로 도면 좌표로 바꿔 전송한다.
    point: Optional[Tuple[float, float]] = None
    # 상황 단위 중복 억제용 (incident.py가 채운다)
    #   'new'        : 새로 시작된 상황 (최초 1회만)
    #   'escalation' : 같은 상황이 오래 지속되어 다시 알리는 것
    stage: str = "new"
    duration: float = 0.0


class TrackManager:
    def __init__(self, frame_height: Optional[int] = None, loiter_sec: float = LOITER_SEC):
        self.tracks: Dict[int, Track] = {}
        self.frame_height = frame_height   # 03 판정에 화면 높이 대비 계산 쓰려면 지정
        self.loiter_sec = loiter_sec       # 05 임계 시간 (테스트 시 짧게 주입 가능)

        # 01 폭행: 카메라 단위 누적 점수 (track_id에 의존하지 않음 - 위 docstring 참고)
        self.assault_score: float = 0.0
        self._assault_score_t: Optional[float] = None
        self.assault_last_info: str = ""   # 디버그 화면 표시용 (마지막으로 점수 오른 근거)

        self._global_last_emit: Dict[str, float] = {}  # code -> 마지막 전송 시각

    # -----------------------------------------------------------------
    def _global_can_emit(self, code: str, now: float) -> bool:
        last = self._global_last_emit.get(code)
        return last is None or (now - last) >= GLOBAL_EVENT_COOLDOWN_SEC

    def _global_mark_emitted(self, code: str, now: float) -> None:
        self._global_last_emit[code] = now

    def update(self, detections: List[Tuple[int, Box]], now: Optional[float] = None) -> None:
        """detections: [(track_id, (x1,y1,x2,y2)), ...] - 트래커가 준 결과"""
        now = now if now is not None else time.time()
        for track_id, box in detections:
            track = self.tracks.setdefault(track_id, Track(track_id=track_id))
            track.push(box, now)

        # 오래 안 보인 트랙 정리 (메모리 누수 방지, 10초 이상 미검출 시 제거)
        stale = [tid for tid, t in self.tracks.items() if now - t.last_seen > 10.0]
        for tid in stale:
            del self.tracks[tid]

    # -----------------------------------------------------------------
    # 03 쓰러짐/응급
    # -----------------------------------------------------------------
    def _check_fall(self, track: Track, now: float) -> Optional[AnomalyEvent]:
        box = track.latest_box()
        if box is None:
            return None
        x1, y1, x2, y2 = box
        w, h = max(1.0, x2 - x1), max(1.0, y2 - y1)
        aspect = w / h

        height_ratio_ok = True
        if self.frame_height:
            height_ratio_ok = (h / self.frame_height) <= FALL_MAX_HEIGHT_RATIO

        is_fall_pose = aspect >= FALL_ASPECT_RATIO and height_ratio_ok

        if is_fall_pose:
            if track.fall_state_since is None:
                track.fall_state_since = now
            sustained = now - track.fall_state_since
            if sustained >= FALL_SUSTAIN_SEC and track.can_emit('03', now):
                track.mark_emitted('03', now)
                confidence = min(95.0, 50.0 + sustained * 5.0)
                return AnomalyEvent(
                    code='03',
                    track_ids=[track.track_id],
                    confidence=round(confidence, 1),
                    detail=(
                        "track %d: 자세 가로/세로 비율 %.2f, %.1f초간 낮은 자세 유지"
                        % (track.track_id, aspect, sustained)
                    ),
                    at=now,
                    point=ground_point(box),
                )
        else:
            track.fall_state_since = None
        return None

    # -----------------------------------------------------------------
    # 01 폭행 (점수 누적 방식 - track ID가 바뀌어도 증거가 유지됨)
    # -----------------------------------------------------------------
    def _check_assault(self, now: float) -> Optional[AnomalyEvent]:
        dt = 0.0
        if self._assault_score_t is not None:
            dt = min(1.0, max(0.0, now - self._assault_score_t))  # 프레임 드랍 시 폭주 방지
        self._assault_score_t = now

        # 1) 이번 프레임에 "붙어 있으면서 격하게 움직이는" 쌍이 있는지 확인
        ids = list(self.tracks.keys())
        best = None   # (점수기준, overlap, gap, motion, pair)

        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                ta, tb = self.tracks[ids[i]], self.tracks[ids[j]]
                ba, bb = ta.latest_box(), tb.latest_box()
                if ba is None or bb is None:
                    continue
                # 두 박스 모두 이번 프레임 근처에서 갱신된 것만 (오래된 박스끼리 비교 방지)
                if now - ta.last_seen > 0.5 or now - tb.last_seen > 0.5:
                    continue

                # 깊이 필터: 겉보기 키가 2배 이상 다르면 서로 다른 거리에 있는 사람
                ha, hb = max(1.0, ba[3] - ba[1]), max(1.0, bb[3] - bb[1])
                if max(ha, hb) / min(ha, hb) > ASSAULT_HEIGHT_RATIO_MAX:
                    continue

                overlap = iou(ba, bb)
                gap = gap_ratio(ba, bb)
                cdist = center_dist_ratio(ba, bb)
                motion = max(ta.motion_intensity(now), tb.motion_intensity(now))

                # 근접 조건: 겹치거나(IOU) 또는 팔 닿는 거리(간격/키)
                near = (overlap >= ASSAULT_IOU_THRESH or gap <= ASSAULT_GAP_RATIO) \
                    and cdist <= ASSAULT_CDIST_RATIO
                if near and motion >= ASSAULT_MOTION_THRESH:
                    # 더 가깝고 더 격한 쌍을 대표로 삼는다
                    rank = motion / (1.0 + gap)
                    if best is None or rank > best[0]:
                        best = (rank, overlap, gap, motion, (ta.track_id, tb.track_id))

        # 2) 조건 만족 구간이면 시간에 비례해 점수 누적, 아니면 감쇠
        if best is not None:
            self.assault_score = min(
                ASSAULT_SCORE_MAX, self.assault_score + dt * ASSAULT_SCORE_PER_SEC
            )
            self.assault_last_info = "iou=%.2f gap=%.2f mot=%.1f score=%.2f" % (
                best[1], best[2], best[3], self.assault_score
            )
        else:
            self.assault_score = max(0.0, self.assault_score - dt * ASSAULT_SCORE_DECAY_PER_SEC)

        # 3) 점수가 임계 초과하면 확정
        if best is not None and self.assault_score >= ASSAULT_SCORE_TRIGGER:
            _, best_overlap, best_gap, best_motion, best_pair = best
            self.assault_score = 0.0  # 확정 후 초기화 (연속 재발화 방지)
            pair = list(best_pair)
            # 두 사람 발 위치의 중간 지점을 사건 위치로 본다
            pair_point = None
            pa = self.tracks.get(best_pair[0]), self.tracks.get(best_pair[1])
            boxes = [t.latest_box() for t in pa if t and t.latest_box()]
            if boxes:
                gps = [ground_point(b) for b in boxes]
                pair_point = (sum(g[0] for g in gps) / len(gps),
                              sum(g[1] for g in gps) / len(gps))
            # 신뢰도: 가까울수록 / 격할수록 높게. 겹침까지 있으면 가산.
            confidence = min(
                92.0,
                50.0
                + max(0.0, (ASSAULT_GAP_RATIO - best_gap)) * 30.0
                + best_overlap * 60.0
                + min(20.0, best_motion * 6.0),
            )
            return AnomalyEvent(
                code='01',
                track_ids=pair,
                confidence=round(confidence, 1),
                detail=(
                    "두 사람이 밀착한 상태에서 격한 몸동작이 반복 감지됨 "
                    "(겹침 %.2f, 간격비 %.2f, 움직임 세기 %.1f)"
                    % (best_overlap, best_gap, best_motion)
                ),
                at=now,
                point=pair_point,
            )
        return None

    # -----------------------------------------------------------------
    # 04 무단침입 (영업시간 외 사람 감지)
    # -----------------------------------------------------------------
    def _check_intrusion(self, now: float, business_hours: Tuple[int, int],
                         force_after_hours: bool = False,
                         wall_now: Optional[float] = None) -> List[AnomalyEvent]:
        open_hour, close_hour = business_hours
        if not force_after_hours:
            # [2026-09-18 수정] 영업시간 판정의 기준 시각을 now와 분리했다.
            #   now는 카메라 모드에선 벽시계지만 영상 파일 모드에선 '영상 재생 위치'(0초부터)다.
            #   그래서 영상으로 테스트하면 1970-01-01 기준으로 시각이 계산돼 04가 오탐됐다.
            #   "지금 매장이 닫혀 있나"는 영상 시간축이 아니라 실제 벽시계로 봐야 하는 값이라
            #   wall_now(기본값 = 실제 현재시각)를 따로 받는다. 테스트에서는 가짜 시각 주입 가능.
            local_hour = time.localtime(wall_now if wall_now is not None else time.time()).tm_hour
            if open_hour <= local_hour < close_hour:
                return []   # 영업시간 중이면 판정 안 함

        events = []
        for track in self.tracks.values():
            if track.can_emit('04', now):
                track.mark_emitted('04', now)
                events.append(AnomalyEvent(
                    code='04',
                    track_ids=[track.track_id],
                    confidence=85.0,
                    detail="track %d: 영업시간(%d~%d시) 외 인원 감지" % (
                        track.track_id, open_hour, close_hour),
                    at=now,
                    point=ground_point(track.latest_box()) if track.latest_box() else None,
                ))
        return events

    # -----------------------------------------------------------------
    # 05 장시간체류
    # -----------------------------------------------------------------
    def _check_loitering(self, now: float) -> List[AnomalyEvent]:
        events = []
        for track in self.tracks.values():
            stayed = track.duration(now)
            if stayed >= self.loiter_sec and track.can_emit('05', now):
                track.mark_emitted('05', now)
                confidence = min(90.0, 60.0 + (stayed - self.loiter_sec) / 60.0 * 5.0)
                events.append(AnomalyEvent(
                    code='05',
                    track_ids=[track.track_id],
                    confidence=round(confidence, 1),
                    detail="track %d: 매장 내 %.1f분간 체류" % (track.track_id, stayed / 60.0),
                    at=now,
                    point=ground_point(track.latest_box()) if track.latest_box() else None,
                ))
        return events

    # -----------------------------------------------------------------
    def check_all(self, now: Optional[float] = None,
                  business_hours: Tuple[int, int] = (9, 22),
                  force_after_hours: bool = False,
                  wall_now: Optional[float] = None) -> List[AnomalyEvent]:
        now = now if now is not None else time.time()
        events: List[AnomalyEvent] = []

        for track in self.tracks.values():
            ev = self._check_fall(track, now)
            if ev:
                events.append(ev)

        assault = self._check_assault(now)
        if assault:
            events.append(assault)

        events.extend(self._check_intrusion(now, business_hours, force_after_hours, wall_now))
        events.extend(self._check_loitering(now))

        # 전역 쿨다운 필터: track ID가 자주 바뀌어도 같은 코드가 연달아 나가는 걸 최종 차단
        filtered = []
        for ev in events:
            if self._global_can_emit(ev.code, now):
                self._global_mark_emitted(ev.code, now)
                filtered.append(ev)
        return filtered
