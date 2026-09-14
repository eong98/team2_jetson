# -*- coding: utf-8 -*-
"""
homography.py

카메라 화면의 픽셀 좌표를 매장 도면(평면도) 좌표로 변환한다.
이슈가 발생했을 때 "매장 어디에서 발생했는지"를 도면 위 X, Y로 보내기 위한 모듈.

[원리]
카메라는 매장 바닥을 비스듬히 내려다보고, 도면은 같은 바닥을 위에서 본 그림이다.
둘 다 '같은 평면(바닥)'을 다른 각도에서 본 것이므로, 두 화면의 좌표는 3x3 행렬 하나로
정확히 변환된다. 이 행렬이 호모그래피(homography)다.

    [x']   [h11 h12 h13]   [x]
    [y'] = [h21 h22 h23] * [y]        (마지막에 w로 나눠줌 - 원근 보정)
    [w ]   [h31 h32 h33]   [1]

대응점(같은 지점을 카메라에서 본 좌표 / 도면에서의 좌표) 4쌍이면 행렬을 구할 수 있다.
(미지수 8개 = 대응점 1쌍당 방정식 2개 x 4쌍. 점이 많을수록 오차가 줄어들어 6~8쌍 권장)

[!!! 가장 중요한 주의점 !!!]
변환은 '바닥 평면 위의 점'에만 성립한다. 바운딩박스의 '중심'을 넣으면 안 된다.
박스 중심은 사람 몸통 높이(공중에 떠 있는 점)라서, 도면에 찍으면 카메라에서 먼 쪽으로
밀려나간 엉뚱한 위치가 나온다. 반드시 **박스 하단 중앙(= 발이 바닥에 닿는 지점)**을 쓸 것.
이 모듈의 box_to_ground_point()가 그 계산을 담당한다.

[사용 방법]
1. 캘리브레이션 도구로 대응점을 찍어 homography.json을 만든다 (카메라마다 1회).
2. 실행 시 이 모듈이 자동으로 읽어서 변환에 사용한다.
3. 카메라 위치나 각도가 바뀌면 반드시 다시 캘리브레이션해야 한다 (행렬이 무효가 됨).

homography.json 형식:
{
  "cno": 1,
  "image_size": [1280, 720],          # 캘리브레이션할 때 쓴 카메라 이미지 크기
  "plan_size": [800, 600],            # 도면 이미지 크기 (정규화 기준이므로 필수)
  "normalize": true,                   # true면 결과를 0~1 비율로 반환 (기본값 true)
  "points": [                          # 대응점 4쌍 이상
    {"camera": [320, 610], "plan": [120, 480]},
    {"camera": [980, 600], "plan": [660, 480]},
    ...
  ]
}

[좌표 단위 - 중요]
서버의 AI 이슈 도면 API(`POST /api/shopmap/issue`)는 xpos/ypos를 **0~1 비율**로만 받는다
(`if not 0 <= xpos <= 1: raise ValueError(...)`). 도면 픽셀 좌표를 그대로 보내면 400 에러가 난다.
그래서 이 모듈은 기본적으로 도면 픽셀을 plan_size로 나눠 0~1로 정규화해서 돌려준다.
도면 픽셀 값 그대로가 필요하면 설정에 "normalize": false 를 넣으면 된다.
"""

import json
import os

import numpy as np

try:
    import cv2
except ImportError:      # cv2 없이 단위 테스트할 때를 위한 방어
    cv2 = None

DEFAULT_CONFIG = "homography.json"

# 도면 경계를 이만큼(비율) 벗어난 것까지는 경계값으로 당겨서 인정한다.
# 0.02 = 도면 크기의 2%. 경계에 선 사람이 부동소수점 오차나 약간의 캘리브레이션 오차로
# 좌표를 통째로 잃는 걸 막기 위한 값.
BOUNDARY_TOLERANCE = 0.02


class Homography:
    """카메라 픽셀 좌표 -> 도면 좌표 변환기."""

    def __init__(self, matrix=None, image_size=None, plan_size=None, normalize=True):
        self.matrix = matrix              # 3x3 numpy array, 없으면 변환 비활성
        self.image_size = image_size      # 캘리브레이션 당시 카메라 해상도 (w, h)
        self.plan_size = plan_size        # 도면 크기 (w, h) - 0~1 정규화 기준
        self.normalize = normalize        # True면 0~1 비율로 반환 (서버 API 요구사항)

    # -----------------------------------------------------------------
    @classmethod
    def load(cls, path=DEFAULT_CONFIG):
        """설정 파일을 읽어 호모그래피 행렬을 계산한다. 파일이 없으면 변환 비활성 상태로 반환."""
        if not os.path.exists(path):
            print("[homography] %s 없음 - 좌표 변환 비활성 (이슈는 좌표 없이 전송됨)" % path)
            return cls()

        try:
            with open(path, "r") as f:
                cfg = json.load(f)

            pts = cfg.get("points", [])
            if len(pts) < 4:
                print("[homography] 대응점이 %d개뿐입니다. 최소 4개 필요 - 변환 비활성" % len(pts))
                return cls()

            src = np.array([p["camera"] for p in pts], dtype=np.float32)
            dst = np.array([p["plan"] for p in pts], dtype=np.float32)

            if cv2 is not None:
                # RANSAC: 잘못 찍힌 대응점 하나가 전체를 망치지 않도록 이상치를 걸러준다
                matrix, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
                used = int(mask.sum()) if mask is not None else len(pts)
                print("[homography] 로드 완료 (대응점 %d개 중 %d개 사용)" % (len(pts), used))
            else:
                matrix = _find_homography_numpy(src, dst)
                print("[homography] 로드 완료 (numpy 계산, 대응점 %d개)" % len(pts))

            if matrix is None:
                print("[homography] 행렬 계산 실패 - 대응점이 한 직선 위에 있지 않은지 확인하세요")
                return cls()

            normalize = cfg.get("normalize", True)
            plan_size = cfg.get("plan_size")
            if normalize and not plan_size:
                print("[homography] normalize=true인데 plan_size가 없습니다 - 정규화 없이 동작")
                normalize = False

            return cls(matrix=np.array(matrix, dtype=np.float64),
                       image_size=cfg.get("image_size"),
                       plan_size=plan_size,
                       normalize=normalize)

        except Exception as e:
            print("[homography] 설정 로드 실패: %s - 좌표 변환 비활성" % e)
            return cls()

    # -----------------------------------------------------------------
    @property
    def enabled(self):
        return self.matrix is not None

    @staticmethod
    def box_to_ground_point(box):
        """바운딩박스에서 '바닥에 닿는 점'(발 위치)을 구한다 = 하단 중앙.

        박스 중심을 쓰면 안 되는 이유는 이 파일 상단 주석 참고.
        """
        x1, y1, x2, y2 = box
        return ((x1 + x2) / 2.0, y2)

    def to_plan(self, point):
        """카메라 픽셀 좌표 (x, y) -> 도면 좌표 (X, Y). 변환 불가면 None."""
        if not self.enabled or point is None:
            return None

        x, y = float(point[0]), float(point[1])
        src = np.array([x, y, 1.0], dtype=np.float64)
        dst = self.matrix.dot(src)

        w = dst[2]
        if abs(w) < 1e-9:        # 지평선 근처(무한대로 발산하는 영역)
            return None

        px, py = float(dst[0] / w), float(dst[1] / w)

        if not self.normalize:
            return (round(px, 1), round(py, 1))

        # 서버 API가 0~1 비율만 받으므로 도면 크기로 나눈다
        nx = px / float(self.plan_size[0])
        ny = py / float(self.plan_size[1])

        # 도면 경계에 딱 걸친 점은 부동소수점 오차로 1.0000001 같은 값이 나온다.
        # 그대로 버리면 매장 가장자리에 선 사람의 좌표가 사라지므로, 살짝 벗어난 건 경계로 당긴다.
        # 반대로 크게 벗어난 건(카메라가 매장 밖까지 비추거나 캘리브레이션이 틀린 경우)
        # 좌표를 안 보내는 게 낫다 - 서버가 400을 내기도 하고, 엉뚱한 위치가 찍히면 더 나쁘다.
        if not (-BOUNDARY_TOLERANCE <= nx <= 1.0 + BOUNDARY_TOLERANCE and
                -BOUNDARY_TOLERANCE <= ny <= 1.0 + BOUNDARY_TOLERANCE):
            return None

        nx = min(1.0, max(0.0, nx))
        ny = min(1.0, max(0.0, ny))
        return (round(nx, 4), round(ny, 4))

    def box_to_plan(self, box):
        """바운딩박스 -> 도면 좌표. 실제로 워커에서 호출하는 건 보통 이 함수."""
        return self.to_plan(self.box_to_ground_point(box))


def _find_homography_numpy(src, dst):
    """cv2가 없을 때 쓰는 최소 구현 (DLT + 최소제곱). 보통은 cv2.findHomography가 쓰인다."""
    n = len(src)
    A = []
    for i in range(n):
        x, y = float(src[i][0]), float(src[i][1])
        u, v = float(dst[i][0]), float(dst[i][1])
        A.append([x, y, 1, 0, 0, 0, -u * x, -u * y, -u])
        A.append([0, 0, 0, x, y, 1, -v * x, -v * y, -v])
    A = np.array(A, dtype=np.float64)
    try:
        _, _, vt = np.linalg.svd(A)
    except np.linalg.LinAlgError:
        return None
    h = vt[-1]
    if abs(h[8]) < 1e-12:
        return None
    return (h / h[8]).reshape(3, 3)


# 전역 인스턴스 (워커에서 import해서 바로 사용)
_instance = None


def get_homography(path=DEFAULT_CONFIG):
    global _instance
    if _instance is None:
        _instance = Homography.load(path)
    return _instance
