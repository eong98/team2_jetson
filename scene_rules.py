# -*- coding: utf-8 -*-
"""
scene_rules.py

02(기물파손) / 06(화재) 판정용 "프레임 단위" CV 휴리스틱.
anomaly_rules.py(01/03/04/05)는 사람 바운딩박스를 보지만, 기물파손·화재는 감지 대상이 사람이
아니라서 화면 전체의 변화를 봐야 한다. 사람 박스는 오탐 방지용으로만 사용(사람 때문에 생긴
변화는 판정에서 제외).

[2026-09-14 기물파손 로직 재작성 - 중요]
  이전 버전은 cv2.createBackgroundSubtractorMOG2()의 전경 마스크 비율만 봤는데, 시뮬레이션으로
  검증해보니 **MOG2가 바뀐 장면을 약 18프레임 만에 배경으로 흡수**해버려서, 4초 지속 조건을
  채우기 전에 전경이 사라져 이벤트가 영원히 발생하지 않았다(미탐).
  원리: MOG2는 "계속 보이는 것 = 배경"으로 학습하는 적응형 모델이라, 넘어진 진열대도 몇 초만
  지나면 정상 배경으로 간주한다. 즉 "바뀐 상태가 유지되는지"를 판단하는 용도로는 부적합.

  그래서 **기준 프레임(reference) 고정 방식**으로 바꿨다:
    1. 평소엔 기준 프레임을 천천히 갱신 (조명 변화 등에 서서히 적응)
    2. 프레임 간 차이가 급증하면(=뭔가 큰 일이 벌어짐) 기준 프레임 갱신을 **얼린다**
       -> 사건 직전의 매장 모습이 기준으로 보존됨
    3. 움직임이 잦아든 뒤, 현재 화면을 "사건 전 기준"과 비교
       - 사람이 그냥 지나간 경우: 지나가고 나면 차이 0 -> 이벤트 없음
       - 물건이 넘어지거나 깨진 경우: 차이가 그대로 남음 -> 지속되면 02 확정
    4. 이벤트 발생 후엔 현재 화면을 새 기준으로 삼아 같은 상황으로 반복 발화하지 않게 함

06 화재: HSV에서 불꽃색(주황/빨강/노랑 계열, 고채도·고명도) 비율 + 그 비율이 프레임마다
출렁이는지(flicker)를 같이 본다. 정지된 빨간 물체(간판/옷)는 flicker가 낮아 걸러지고, 실제
불꽃은 흔들려서 걸린다. 편의점 특성상 빨간 상품이 많아 오탐 위험 있음 - 실측 튜닝 필요.

Python 3.6 환경(Jetson Nano) 기준으로 작성.
"""

from collections import deque
from typing import Deque, List, Optional, Tuple

import cv2
import numpy as np

from anomaly_rules import AnomalyEvent

Box = Tuple[float, float, float, float]

# ---------------------------------------------------------------------------
# 02 기물파손 임계값 (전부 튜닝 대상)
# ---------------------------------------------------------------------------
DAMAGE_PIXEL_DIFF = 30            # 픽셀 밝기 차이가 이 값을 넘으면 "달라진 픽셀"로 카운트
DAMAGE_QUIET_RATIO = 0.02         # 직전 프레임 대비 움직임이 이 아래면 "상황이 잦아들었다"
# [2026-09-14 변경] 사건 감지 기준을 "직전 프레임 대비 순간 움직임"에서 "기준 대비 변화량의 급상승"으로 교체.
#   합성 테스트 영상으로 확인해보니, 진열대가 넘어지는 0.6초 동안의 순간 움직임은 화면의 3.8%뿐이라
#   기존 6% 임계를 못 넘어 판정 자체가 시작되지 않았다(미탐). 진열대 하나는 화면에서 작기 때문.
#   그래서 "최근 2초 사이에 기준 대비 변화량이 얼마나 빠르게 늘었는가"를 본다.
#   조명이 서서히 변하는 것과 물건이 갑자기 넘어지는 것을 구분하는 건 '변화량'이 아니라 '변화 속도'다.
DAMAGE_RISE_RATIO = 0.012         # 최근 2초간 기준 대비 변화량이 이만큼 급증하면 "사건 발생"
DAMAGE_RISE_WINDOW_SEC = 2.0
DAMAGE_BG_DIFF_RATIO = 0.03       # 사건 전 기준과 비교해 이 비율 이상 달라져 있으면 "뭔가 바뀐 채로 있음"
# [2026-09-14 추가] 변화가 이 비율을 넘으면 기물파손이 아니라 '장면 자체가 바뀐 것'으로 본다.
#   실기기 테스트에서, 모니터에 재생 중인 영상을 카메라로 찍다가 영상의 장면이 전환되자
#   화면의 44%/60%가 한꺼번에 바뀌면서 02가 오탐으로 발생했다.
#   고정 CCTV에서 진열대가 넘어져도 화면의 5~20% 수준이 바뀔 뿐, 절반이 통째로 바뀌지는 않는다.
#   그런 경우는 카메라가 움직였거나 장면이 교체된 것이므로, 이벤트를 내지 말고 기준을 새로 잡는다.
DAMAGE_MAX_DIFF_RATIO = 0.35
DAMAGE_SUSTAIN_SEC = 3.0          # 위 상태가 이만큼(초) 유지돼야 확정
DAMAGE_JUDGE_WINDOW_SEC = 30.0    # 스파이크 후 이 시간 안에서만 판정 (넘으면 기준 프레임 다시 갱신)
DAMAGE_REF_ALPHA = 0.05           # 평상시 기준 프레임 갱신 속도 (클수록 조명 변화에 빨리 적응)
DAMAGE_COOLDOWN_SEC = 60.0
DAMAGE_WARMUP_SEC = 3.0           # 시작 직후 이 시간 동안은 판정 보류 (기준 프레임 안정화)
DAMAGE_PERSON_MARGIN = 20         # 사람 박스를 이만큼 px 넓혀서 제외 (그림자/잔상 여유)

# ---------------------------------------------------------------------------
# 06 화재 임계값
# ---------------------------------------------------------------------------
# [2026-09-14 오탐 수정] 실기기에서 불이 없는데도 "불꽃색 3.4%, 깜빡임 1.71"로 06이 발생했다.
#   원인 1: 화면 전체의 불꽃색 '픽셀 수'만 셌기 때문. 매장의 나무 테이블, 베이지색 상품,
#           따뜻한 조명에 흩어져 있는 주황 픽셀이 다 합산돼서 임계를 넘었다.
#           -> 실제 불은 '한 덩어리'로 뭉쳐 있다. 흩어진 픽셀 말고 **가장 큰 덩어리 크기**를 본다.
#   원인 2: flicker = 표준편차/평균 이라 평균이 작으면 값이 폭증한다. 임계에 아슬아슬하게
#           걸친 소량의 픽셀이 카메라 자동노출 때문에 들락날락하면 flicker가 1.7까지 뛴다.
#           -> 실제 불꽃의 flicker는 0.2~0.6 수준. 너무 큰 값은 오히려 노이즈이므로 상한을 둔다.
FIRE_MIN_BLOB_RATIO = 0.012       # 가장 큰 불꽃색 '덩어리'가 화면의 이 비율 이상이어야 함
FIRE_MAX_BLOB_RATIO = 0.45        # 이보다 크면 불이 아니라 조명/석양/화면 전체 색조 변화
FIRE_FLICKER_MIN_STD = 0.12       # 이 아래면 정지된 주황 물체(간판/옷/나무)
FIRE_FLICKER_MAX_STD = 1.10       # 이 위는 실제 불꽃이 아니라 노이즈/자동노출 요동
FIRE_WINDOW_SEC = 5.0             # flicker 계산에 쓰는 시간창
FIRE_SUSTAIN_SEC = 3.0
FIRE_COOLDOWN_SEC = 60.0
FIRE_ANALYZE_WIDTH = 640          # 덩어리 분석은 축소해서 (젯슨 부하 절감, 정확도 영향 미미)


class DamageDetector:
    """기준 프레임 고정 방식의 기물파손(02) 검출기. 위 docstring의 4단계 설명 참고."""

    def __init__(self):
        self._ref = None             # 기준 프레임 (float32 그레이스케일)
        self._prev_gray = None       # 직전 프레임 (스파이크 계산용)
        self._spike_at = None        # 마지막 스파이크 시각
        self._changed_since = None   # "바뀐 채로 유지" 시작 시각
        self._last_emit_at = None
        self._started_at = None
        self._diff_history = deque(maxlen=300)   # (시각, 기준 대비 변화량) - 급상승 판단용

    @staticmethod
    def _to_gray(frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # 노이즈(카메라 지글거림)로 오탐 나지 않게 살짝 블러
        return cv2.GaussianBlur(gray, (5, 5), 0)

    @staticmethod
    def _mask_centroid(mask):
        """마스크(변화가 감지된 픽셀)의 무게중심 = 사건이 일어난 화면상 위치.
        도면 좌표 변환에 쓰인다. 바닥에 쏟아진 물건의 중심이므로 바닥 평면으로 봐도 무리 없음."""
        ys, xs = np.nonzero(mask)
        if len(xs) == 0:
            return None
        return (float(xs.mean()), float(ys.mean()))

    @staticmethod
    def _mask_people(mask, person_boxes: List[Box]):
        h, w = mask.shape[:2]
        m = DAMAGE_PERSON_MARGIN
        for (x1, y1, x2, y2) in person_boxes:
            ax1, ay1 = max(0, int(x1) - m), max(0, int(y1) - m)
            ax2, ay2 = min(w, int(x2) + m), min(h, int(y2) + m)
            if ax2 > ax1 and ay2 > ay1:
                mask[ay1:ay2, ax1:ax2] = 0
        return mask

    def update(self, frame, person_boxes: List[Box], now: float):
        """반환: (이벤트 또는 None, 기준 대비 변화 비율) - 뒤 값은 디버그 화면 표시용."""
        gray = self._to_gray(frame)

        if self._ref is None:
            self._ref = gray.astype(np.float32)
            self._prev_gray = gray
            self._started_at = now
            return None, 0.0

        # 1) 직전 프레임 대비 움직임(스파이크 판단용)
        motion_mask = (cv2.absdiff(gray, self._prev_gray) > DAMAGE_PIXEL_DIFF).astype(np.uint8)
        motion_mask = self._mask_people(motion_mask, person_boxes)
        motion_ratio = float(np.count_nonzero(motion_mask)) / motion_mask.size
        self._prev_gray = gray

        # 2) 기준 프레임 대비 변화(=지금 뭔가 바뀐 채로 있는지)
        diff_mask = (cv2.absdiff(gray, self._ref.astype(np.uint8)) > DAMAGE_PIXEL_DIFF).astype(np.uint8)
        diff_mask = self._mask_people(diff_mask, person_boxes)
        diff_ratio = float(np.count_nonzero(diff_mask)) / diff_mask.size

        warming_up = (now - self._started_at) < DAMAGE_WARMUP_SEC
        self._diff_history.append((now, diff_ratio))

        # 2-b) 화면이 통째로 바뀐 경우(카메라 이동/장면 전환)는 기물파손이 아니다.
        #      이벤트를 내지 않고 현재 화면을 새 기준으로 삼는다.
        if diff_ratio >= DAMAGE_MAX_DIFF_RATIO:
            self._ref = gray.astype(np.float32)
            self._changed_since = None
            self._spike_at = None
            return None, diff_ratio

        # 3) 사건 감지: 최근 DAMAGE_RISE_WINDOW_SEC 사이에 기준 대비 변화량이 급증했는가
        recent = [r for t, r in self._diff_history if now - t <= DAMAGE_RISE_WINDOW_SEC]
        if not warming_up and len(recent) >= 2:
            if (diff_ratio - min(recent)) >= DAMAGE_RISE_RATIO:
                self._spike_at = now      # 기준 프레임을 얼려서 "사건 직전 모습"을 보존

        judging = self._spike_at is not None and (now - self._spike_at) <= DAMAGE_JUDGE_WINDOW_SEC

        # 4) 기준 프레임 갱신은 "지금 화면이 기준과 거의 같을 때"만 한다.
        #    [중요] 예전엔 판정 중이 아니기만 하면 갱신했는데, 그러면 넘어진 진열대가 몇 초 만에
        #    기준 프레임에 서서히 흡수돼서(변화량 5.3% -> 0%) 영원히 이벤트가 안 뜬다.
        #    MOG2를 걷어낸 이유와 똑같은 함정이라, 갱신 조건을 "정상 상태일 때만"으로 좁혔다.
        if (not judging) and motion_ratio < DAMAGE_QUIET_RATIO and diff_ratio < DAMAGE_BG_DIFF_RATIO:
            cv2.accumulateWeighted(gray, self._ref, DAMAGE_REF_ALPHA)

        if warming_up:
            return None, diff_ratio

        # 5) 스파이크가 있었고 + 움직임이 잦아들었고 + 기준과 여전히 다르면 -> 지속시간 누적
        settled = motion_ratio < DAMAGE_QUIET_RATIO
        if judging and settled and diff_ratio >= DAMAGE_BG_DIFF_RATIO:
            if self._changed_since is None:
                self._changed_since = now
            sustained = now - self._changed_since

            can_emit = self._last_emit_at is None or (now - self._last_emit_at) >= DAMAGE_COOLDOWN_SEC
            if sustained >= DAMAGE_SUSTAIN_SEC and can_emit:
                self._last_emit_at = now
                self._changed_since = None
                self._spike_at = None
                self._ref = gray.astype(np.float32)   # 새 상태를 기준으로 재설정(반복 발화 방지)
                confidence = min(80.0, 40.0 + diff_ratio * 200)
                return AnomalyEvent(
                    code='02',
                    track_ids=[],
                    confidence=round(confidence, 1),
                    detail=(
                        "화면에 급격한 변화가 발생한 뒤 매장 모습이 %.1f%% 달라진 상태로 "
                        "%.1f초간 유지됨 (진열대/물품 파손·전도 의심)" % (diff_ratio * 100, sustained)
                    ),
                    at=now,
                    point=self._mask_centroid(diff_mask),
                ), diff_ratio
        else:
            self._changed_since = None

        # 판정 시간이 지났는데 이벤트가 안 났으면(=원상복구됨) 기준 프레임 갱신 재개
        if self._spike_at is not None and (now - self._spike_at) > DAMAGE_JUDGE_WINDOW_SEC:
            self._spike_at = None

        return None, diff_ratio


class FireDetector:
    """HSV 색상 + 깜빡임(flicker) 기반 화재(06) 검출기."""

    # 불꽃색 범위(HSV): 빨강~주황~노랑. 채도/명도 하한을 올려서 '밝고 진한' 것만 남긴다.
    # (기존 S>=100, V>=150 은 매장의 나무/베이지 계열까지 통과시켜 오탐의 원인이 됐다)
    _LOWER = np.array([0, 130, 190], dtype=np.uint8)
    _UPPER = np.array([35, 255, 255], dtype=np.uint8)

    def __init__(self):
        self._ratio_history: Deque[Tuple[float, float]] = deque(maxlen=300)
        self._fire_since: Optional[float] = None
        self._last_emit_at: Optional[float] = None
        self._last_mask = None
        self._last_scale = 1.0

    @staticmethod
    def _fire_base_point(mask):
        """불꽃 영역의 '아래쪽 중심'을 발화 지점으로 본다.
        불은 위로 솟으므로 영역 전체의 무게중심을 쓰면 실제 발화점보다 위로 뜬다.
        바닥과 만나는 지점을 잡아야 도면 좌표가 맞는다."""
        if mask is None:
            return None
        ys, xs = np.nonzero(mask)
        if len(xs) == 0:
            return None
        bottom = ys.max()
        near_bottom = ys >= (bottom - max(5, (bottom - ys.min()) * 0.15))
        return (float(xs[near_bottom].mean()), float(ys[near_bottom].mean()))

    def _fire_point_original(self):
        """축소해서 분석했으므로, 좌표를 원본 프레임 기준으로 되돌린다."""
        p = self._fire_base_point(self._last_mask)
        if p is None or self._last_scale in (0, 1.0):
            return p
        return (p[0] / self._last_scale, p[1] / self._last_scale)

    def _largest_blob(self, mask):
        """가장 큰 불꽃색 덩어리의 (면적비율, 마스크)를 구한다.

        흩어진 주황 픽셀(상품/조명)과 뭉쳐 있는 불꽃을 가르는 핵심 단계.
        작은 점들은 열림 연산으로 지우고, 연결된 덩어리 중 가장 큰 것만 남긴다.
        """
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        if n <= 1:
            return 0.0, None
        # 0번은 배경이므로 1번부터
        areas = stats[1:, cv2.CC_STAT_AREA]
        idx = int(np.argmax(areas)) + 1
        blob = (labels == idx).astype(np.uint8) * 255
        return float(areas[idx - 1]) / mask.size, blob

    def update(self, frame, now: float):
        """반환: (이벤트 또는 None, 가장 큰 불꽃색 덩어리 비율, flicker 지수)"""
        h, w = frame.shape[:2]
        if w > FIRE_ANALYZE_WIDTH:
            scale = FIRE_ANALYZE_WIDTH / float(w)
            small = cv2.resize(frame, (FIRE_ANALYZE_WIDTH, int(h * scale)))
        else:
            scale, small = 1.0, frame

        hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self._LOWER, self._UPPER)

        ratio, blob = self._largest_blob(mask)   # 전체 픽셀 수가 아니라 '가장 큰 덩어리'
        self._ratio_history.append((now, ratio))
        self._last_mask = blob
        self._last_scale = scale

        window = [r for t, r in self._ratio_history if now - t <= FIRE_WINDOW_SEC]
        if len(window) < 4:
            return None, ratio, 0.0

        mean_ratio = sum(window) / len(window)
        variance = sum((r - mean_ratio) ** 2 for r in window) / len(window)
        flicker = (variance ** 0.5 / mean_ratio) if mean_ratio > 1e-6 else 0.0

        is_fire_like = (
            FIRE_MIN_BLOB_RATIO <= mean_ratio <= FIRE_MAX_BLOB_RATIO
            and FIRE_FLICKER_MIN_STD <= flicker <= FIRE_FLICKER_MAX_STD
        )

        if is_fire_like:
            if self._fire_since is None:
                self._fire_since = now
            sustained = now - self._fire_since
            can_emit = self._last_emit_at is None or (now - self._last_emit_at) >= FIRE_COOLDOWN_SEC
            if sustained >= FIRE_SUSTAIN_SEC and can_emit:
                self._last_emit_at = now
                self._fire_since = None
                confidence = min(75.0, 25.0 + mean_ratio * 200 + flicker * 50)
                return AnomalyEvent(
                    code='06',
                    track_ids=[],
                    confidence=round(confidence, 1),
                    detail=(
                        "불꽃으로 의심되는 영역(화면의 %.1f%%)이 깜빡임 지수 %.2f로 "
                        "%.1f초간 지속됨" % (mean_ratio * 100, flicker, sustained)
                    ),
                    at=now,
                    point=self._fire_point_original(),
                ), ratio, flicker
        else:
            self._fire_since = None

        return None, ratio, flicker
