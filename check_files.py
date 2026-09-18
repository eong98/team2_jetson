# -*- coding: utf-8 -*-
"""
check_files.py

젯슨의 워커 파일들이 서로 짝이 맞는지 확인한다.
파일을 하나씩 pscp로 올리다 보면 일부만 갱신돼서 "update() takes 3 positional arguments but
4 were given" 같은 시그니처 불일치가 생긴다. 워커를 돌리기 전에 이걸 먼저 실행하면
카메라를 열기도 전에 문제를 알 수 있다.

    python3 check_files.py
"""

import os
import sys

REQUIRED = [
    "worker_skeleton.py",
    "anomaly_rules.py",
    "scene_rules.py",
    "person_detector.py",
    "incident.py",
    "visitor.py",
    "homography.py",
]

print("=== 파일 존재 확인 ===")
missing = [f for f in REQUIRED if not os.path.exists(f)]
for f in REQUIRED:
    print("  %-22s %s" % (f, "있음" if os.path.exists(f) else "없음 <-- 올려야 함"))
if missing:
    print("\n필수 파일이 없습니다: %s" % ", ".join(missing))
    sys.exit(1)

print("\n=== 모듈 임포트 + 시그니처 확인 ===")
try:
    import inspect
    from scene_rules import DamageDetector, FireDetector
    from anomaly_rules import TrackManager, AnomalyEvent
    from incident import IncidentTracker
    from visitor import VisitorTracker
    from homography import get_homography

    checks = [
        ("DamageDetector.update", DamageDetector.update, 4),   # self, frame, person_boxes, now
        ("FireDetector.update", FireDetector.update, 4),
    ]
    bad = []
    for name, fn, expect in checks:
        n = len(inspect.getfullargspec(fn).args)
        ok = n == expect
        print("  %-24s 인자 %d개 %s" % (name, n, "OK" if ok else "<-- 파일 버전 불일치"))
        if not ok:
            bad.append(name)

    # AnomalyEvent에 최신 필드가 있는지 (stage/duration/point)
    fields = getattr(AnomalyEvent, "__dataclass_fields__", {})
    for f in ("point", "stage", "duration"):
        ok = f in fields
        print("  %-24s %s" % ("AnomalyEvent." + f, "OK" if ok else "<-- anomaly_rules.py가 구버전"))
        if not ok:
            bad.append("AnomalyEvent." + f)

    # incident.py 최신 기능
    for attr in ("_max_sec", "_blocked_by"):
        ok = hasattr(IncidentTracker, attr)
        print("  %-24s %s" % ("IncidentTracker." + attr, "OK" if ok else "<-- incident.py가 구버전"))
        if not ok:
            bad.append("IncidentTracker." + attr)

    # [2026-09-18] 폭행 근접판정 개편 + 영업시간 기준시각 분리
    import anomaly_rules as _AR
    for attr in ("ASSAULT_GAP_RATIO", "ASSAULT_SCORE_PER_SEC", "gap_ratio", "center_dist_ratio"):
        ok = hasattr(_AR, attr)
        print("  %-24s %s" % ("anomaly_rules." + attr, "OK" if ok else "<-- anomaly_rules.py가 구버전"))
        if not ok:
            bad.append("anomaly_rules." + attr)
    ok = "wall_now" in inspect.getfullargspec(TrackManager.check_all).args
    print("  %-24s %s" % ("check_all(wall_now=)", "OK" if ok else "<-- anomaly_rules.py가 구버전"))
    if not ok:
        bad.append("check_all(wall_now)")
    ok = hasattr(DamageDetector, "_trail_boxes")
    print("  %-24s %s" % ("DamageDetector._trail_boxes", "OK" if ok else "<-- scene_rules.py가 구버전"))
    if not ok:
        bad.append("DamageDetector._trail_boxes")

    print()
    if bad:
        print("불일치 발견: %s" % ", ".join(bad))
        print("-> 워커 파일 전체를 다시 올리세요 (일부만 갱신된 상태입니다)")
        sys.exit(1)

    print("모든 파일이 짝이 맞습니다. 워커를 실행해도 됩니다.")

except ImportError as e:
    print("  임포트 실패: %s" % e)
    print("  -> yolov5 저장소 폴더(~/yolov5) 안에서 실행하고 있는지 확인하세요.")
    sys.exit(1)
