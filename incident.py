# -*- coding: utf-8 -*-
"""
incident.py

같은 이상 상황이 계속 감지될 때 DB와 알림이 폭주하지 않도록, 연속된 감지를
하나의 "상황(incident)"으로 묶어주는 모듈.

[왜 단순 쿨다운이 아니라 이 방식인가]
쿨다운(예: 60초마다 한 번씩만 전송)은 문제를 줄여줄 뿐 없애지 못한다.
쓰러진 사람이 3시간 방치되면 60초 쿨다운으로도 180건이 쌓이고, 문자도 180번 나간다.

근본 원인은 쓰러짐·화재를 '순간 이벤트'로 다루고 있다는 점이다. 이것들은 실제로는
**시작되고 → 지속되다 → 끝나는 '상황'**이다. 그래서 상황 단위로 묶는다:

    최초 감지        -> 전송 O  (CCTV_ISSUE 1건 INSERT + 알림 1회)
    지속되는 동안     -> 전송 X  (계속 감지되고 있지만 이미 아는 상황)
    일정 시간 미감지  -> 상황 종료
    다시 감지        -> 새 상황으로 전송 O

결과: 3시간 쓰러져 있어도 1건. 상황이 실제로 끝나고 다시 발생해야 2건이 된다.

[에스컬레이션 - 그냥 묵살하면 안 되는 이유]
쓰러짐이 10분째 지속된다는 건 아무도 발견하지 못했다는 뜻이라 오히려 더 위급하다.
그래서 무한 반복은 막되, 정해진 시점에만 딱 한 번씩 다시 알린다.

    5분 경과   -> 1차 에스컬레이션 (한 번만)
    15분 경과  -> 2차 에스컬레이션 (한 번만)

즉 3시간 지속되어도 총 3건(최초 + 5분 + 15분)에서 멈춘다. 코드마다 다르게 줄 수 있어서
화재처럼 급한 건 짧게, 장시간체류처럼 덜 급한 건 아예 에스컬레이션을 끄면 된다.

[묶는 기준을 track_id가 아니라 code로 잡은 이유]
젯슨의 낮은 FPS에서는 track_id가 중간에 바뀔 수 있는데(트래커 한계), track_id로 묶으면
같은 사람이 ID가 바뀔 때마다 새 상황으로 잡혀서 중복 억제가 무력화된다.
"이 카메라에서 쓰러짐 상황이 진행 중인가"를 코드 단위로 보는 게 훨씬 안정적이다.
"""

import time

# 상황이 끝났다고 보기까지, 해당 코드가 한 번도 감지되지 않아야 하는 시간(초).
# 짧으면 깜빡임(탐지가 한두 프레임 끊김)에 새 상황으로 잡히고, 길면 실제로 끝난 뒤
# 다시 발생한 상황을 놓친다.
DEFAULT_CLEAR_SEC = 120.0

# 상황 지속 시 재알림할 시점(초). 각 시점마다 딱 한 번씩만 전송한다.
DEFAULT_ESCALATE_AT = (300.0, 900.0)      # 5분, 15분

# 코드별 설정 (없으면 위 기본값 사용)
CODE_POLICY = {
    "01": {"clear_sec": 90.0,  "escalate_at": (180.0, 600.0)},   # 폭행: 빨리 다시 알림
    "02": {"clear_sec": 300.0, "escalate_at": ()},               # 기물파손: 이미 벌어진 일, 재알림 불필요
    "03": {"clear_sec": 120.0, "escalate_at": (300.0, 900.0)},   # 쓰러짐: 방치될수록 위급
    "04": {"clear_sec": 180.0, "escalate_at": (600.0,)},         # 무단침입
    "05": {"clear_sec": 600.0, "escalate_at": ()},               # 장시간체류: 급하지 않음
    "06": {"clear_sec": 60.0,  "escalate_at": (120.0, 300.0)},   # 화재: 가장 급함
}


class Incident(object):
    """진행 중인 상황 하나."""

    def __init__(self, code, started_at):
        self.code = code
        self.started_at = started_at
        self.last_seen = started_at
        self.sent_count = 1               # 최초 전송 1회 포함
        self.escalated = set()            # 이미 보낸 에스컬레이션 시점들
        self.max_confidence = 0.0

    def duration(self, now):
        return now - self.started_at


class IncidentTracker(object):
    """코드별로 진행 중인 상황을 관리하면서, 실제로 전송할 이벤트만 걸러준다."""

    def __init__(self, policy=None):
        self.policy = policy or CODE_POLICY
        self.active = {}                  # code -> Incident

    # -----------------------------------------------------------------
    def _conf(self, code, key, default):
        return self.policy.get(code, {}).get(key, default)

    def _clear_sec(self, code):
        return self._conf(code, "clear_sec", DEFAULT_CLEAR_SEC)

    def _escalate_at(self, code):
        return self._conf(code, "escalate_at", DEFAULT_ESCALATE_AT)

    # -----------------------------------------------------------------
    def process(self, events, now=None):
        """룰 엔진이 만든 이벤트 목록을 받아, 실제로 서버에 보낼 것만 돌려준다.

        이벤트가 없어도 매 프레임 호출해야 한다(상황 종료 판정을 여기서 하기 때문).

        반환된 이벤트에는 아래 두 값이 채워진다:
          event.stage    - 'new'(최초 발생) 또는 'escalation'(지속 중 재알림)
          event.duration - 상황이 시작된 뒤 흐른 시간(초)
        """
        now = now if now is not None else time.time()
        to_send = []

        # 1) 이번에 감지된 코드들을 반영
        seen_codes = set()
        for ev in events:
            code = ev.code
            seen_codes.add(code)
            incident = self.active.get(code)

            if incident is None:
                # 새 상황 시작 -> 최초 1회 전송
                incident = Incident(code, now)
                incident.max_confidence = ev.confidence
                self.active[code] = incident
                ev.stage = "new"
                ev.duration = 0.0
                to_send.append(ev)
            else:
                # 이미 진행 중인 상황 -> 기본적으로 전송하지 않음
                incident.last_seen = now
                incident.max_confidence = max(incident.max_confidence, ev.confidence)

        # 2) 진행 중인 상황들의 에스컬레이션 / 종료 판정
        for code in list(self.active.keys()):
            incident = self.active[code]

            # 이번 프레임에 감지되지 않았고, 유예 시간도 지났으면 상황 종료
            if code not in seen_codes and (now - incident.last_seen) >= self._clear_sec(code):
                del self.active[code]
                continue

            # 아직 진행 중이면 에스컬레이션 시점을 지났는지 확인
            if code not in seen_codes:
                continue      # 잠깐 안 보이는 중에는 재알림하지 않음

            elapsed = incident.duration(now)
            for mark in self._escalate_at(code):
                if elapsed >= mark and mark not in incident.escalated:
                    incident.escalated.add(mark)
                    incident.sent_count += 1
                    base = next((e for e in events if e.code == code), None)
                    if base is not None:
                        esc = self._make_escalation(base, incident, elapsed, mark)
                        to_send.append(esc)

        return to_send

    # -----------------------------------------------------------------
    @staticmethod
    def _make_escalation(base, incident, elapsed, mark):
        """지속 알림용 이벤트를 만든다. 원본을 복사해서 문구와 표시만 바꾼다."""
        import copy
        esc = copy.copy(base)
        esc.stage = "escalation"
        esc.duration = elapsed
        minutes = int(round(elapsed / 60.0))
        esc.detail = "[%d분째 지속 - 미조치] %s" % (minutes, base.detail)
        # 오래 방치될수록 신뢰도(=위험도)를 조금 올려서 보낸다
        esc.confidence = min(99.0, max(base.confidence, incident.max_confidence) + 5.0)
        return esc

    # -----------------------------------------------------------------
    def status_line(self):
        """디버그 화면/로그에 현재 진행 중인 상황을 한 줄로 표시."""
        if not self.active:
            return ""
        now = time.time()
        parts = []
        for code, inc in sorted(self.active.items()):
            parts.append("%s(%.0fs)" % (code, inc.duration(now)))
        return "진행중: " + " ".join(parts)
