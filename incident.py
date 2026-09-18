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

# [2026-09-18 추가 - 중요] 상황의 최대 지속 시간. 이 시간이 지나면 강제로 종료하고,
# 다음 감지는 새 상황으로 취급한다.
#
# 왜 필요한가: clear_sec은 "이 시간 동안 감지가 없어야 종료"인데, 판정기가 자기 쿨다운마다
# 재감지를 보내면 그 공백이 영원히 생기지 않는다. 실제로 02(기물파손)에서 이 일이 터졌다 —
# DamageDetector 쿨다운 60초 < 02의 clear_sec 300초 라서, 60초마다 들어오는 재감지가
# last_seen을 계속 갱신해 상황이 **영구히 진행중**으로 갇혔다. 02는 재알림 정책도 없어서
# 30분 동안 단 1건만 전송되고 그 뒤 모든 기물파손이 묵살됐다(실측 확인).
#
# 즉 "지속 중이니 보내지 않는다"는 억제가, 무한정 이어지면 "새 사건도 못 본다"로 변질된다.
# 상한을 두면 최악의 경우에도 이 주기로는 반드시 다시 보고된다 = 억제의 안전장치.
DEFAULT_MAX_INCIDENT_SEC = 900.0          # 15분

# [2026-09-18 추가] 코드 간 배타 규칙 - {억제될 코드: ((억제하는 코드들), 여유시간초)}
#
# 왜 필요한가: 실기기 테스트에서 02(기물파손)와 03(쓰러짐)이 같이 발생했다.
# 사람이 쓰러지면 (1) 사람 박스가 눕고 -> 03, (2) 화면에도 큰 변화가 생겨 그대로 유지됨 -> 02.
# DamageDetector가 사람 박스 영역을 제외하긴 하지만, 쓰러진 사람을 YOLO가 한동안 놓치거나
# 그림자·주변 물건까지 바뀌면 박스 밖에서 변화가 잡힌다.
#
# 원칙: **사람 때문에 생긴 화면 변화를 기물파손으로 또 보고하지 않는다.**
# 03(쓰러짐)이나 01(폭행) 상황이 진행 중이면 02는 억제한다. 그 상황이 끝난 뒤에도 잠깐은
# 잔상(사람이 있던 자리)이 남으므로 여유시간을 둔다.
# 진짜 기물파손은 사람이 없거나 이미 자리를 떠난 뒤에도 변화가 남아 있으므로 정상적으로 잡힌다.
SUPPRESSED_BY = {
    "02": (("01", "03"), 20.0),
}

# 코드별 설정 (없으면 위 기본값 사용)
# max_sec 규칙: 반드시 해당 판정기의 쿨다운보다 넉넉히 크게, 그리고 clear_sec보다 크게 잡는다.
# (DamageDetector/FireDetector 쿨다운 60초, Track.EVENT_COOLDOWN_SEC 60초)
CODE_POLICY = {
    "01": {"clear_sec": 90.0,  "escalate_at": (180.0, 600.0),  "max_sec": 600.0},   # 폭행
    "02": {"clear_sec": 120.0, "escalate_at": (),              "max_sec": 180.0},   # 기물파손: 순간 사건이라 짧게
    "03": {"clear_sec": 120.0, "escalate_at": (300.0, 900.0),  "max_sec": 1800.0},  # 쓰러짐: 방치될수록 위급
    "04": {"clear_sec": 180.0, "escalate_at": (600.0,),        "max_sec": 1800.0},  # 무단침입
    "05": {"clear_sec": 600.0, "escalate_at": (),              "max_sec": 1800.0},  # 장시간체류
    "06": {"clear_sec": 60.0,  "escalate_at": (120.0, 300.0),  "max_sec": 600.0},   # 화재: 가장 급함
}


class Incident(object):
    """진행 중인 상황 하나."""

    def __init__(self, code, started_at):
        self.code = code
        self.started_at = started_at
        self.last_seen = started_at
        self.sent_count = 1               # 최초 전송 1회 포함
        self.absorbed = 0                 # 같은 상황으로 흡수(=전송 안 함)된 감지 횟수
        self.escalated = set()            # 이미 보낸 에스컬레이션 시점들
        self.max_confidence = 0.0

    def duration(self, now):
        return now - self.started_at


class IncidentTracker(object):
    """코드별로 진행 중인 상황을 관리하면서, 실제로 전송할 이벤트만 걸러준다."""

    def __init__(self, policy=None):
        self.policy = policy or CODE_POLICY
        self.active = {}                  # code -> Incident
        self._ended_at = {}               # code -> 상황이 끝난 시각 (배타 규칙의 여유시간 계산용)
        self._suppress_logged = set()     # 억제 로그를 코드당 한 번만 찍기 위한 표시

    # -----------------------------------------------------------------
    def _conf(self, code, key, default):
        return self.policy.get(code, {}).get(key, default)

    def _clear_sec(self, code):
        return self._conf(code, "clear_sec", DEFAULT_CLEAR_SEC)

    def _escalate_at(self, code):
        return self._conf(code, "escalate_at", DEFAULT_ESCALATE_AT)

    def _max_sec(self, code):
        return self._conf(code, "max_sec", DEFAULT_MAX_INCIDENT_SEC)

    def _blocked_by(self, code, now, incoming=()):
        """이 코드가 다른 상황 때문에 억제되어야 하면 그 코드를, 아니면 None을 반환.

        incoming: 이번 프레임에 같이 들어온 코드들. 첫 프레임에 03과 02가 동시에 도착하는 경우,
        아직 self.active에 아무것도 없으므로 이것까지 봐야 억제가 걸린다.
        """
        rule = SUPPRESSED_BY.get(code)
        if not rule:
            return None
        blockers, grace = rule
        for b in blockers:
            if b in incoming:                 # 같은 프레임에 함께 감지됨
                return b
            if b in self.active:              # 그 상황이 지금 진행 중
                return b
            ended = self._ended_at.get(b)     # 막 끝났고 아직 여유시간 안
            if ended is not None and (now - ended) <= grace:
                return b
        return None

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

        # 0) 배타 규칙: 사람 때문에 생긴 화면 변화를 기물파손으로 중복 보고하지 않는다
        incoming = set(e.code for e in events)
        kept = []
        for ev in events:
            blocker = self._blocked_by(ev.code, now, incoming)
            if blocker is None:
                kept.append(ev)
                self._suppress_logged.discard(ev.code)
            elif ev.code not in self._suppress_logged:
                self._suppress_logged.add(ev.code)
                print("[incident] code=%s 억제 - code=%s 상황이 진행 중이라 같은 원인으로 판단"
                      % (ev.code, blocker))
        events = kept

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
                incident.absorbed += 1
                # [중요] 흡수될 때 아무 로그가 없으면 "탐지가 안 된 것"과 구분이 안 된다.
                # 실제로 01 폭행을 반복 테스트하다가 "갑자기 인식이 안 된다"고 오해한 사례가 있었다.
                # 상황당 첫 흡수 때 한 번만 찍어서 스팸은 피하고 원인은 보이게 한다.
                if incident.absorbed == 1:
                    print("[incident] code=%s 감지됨 - 이미 진행 중인 상황이라 전송 생략 "
                          "(%.0f초 동안 미감지 상태가 되면 종료 후 새 건으로 보고)"
                          % (code, self._clear_sec(code)))

        # 2) 진행 중인 상황들의 에스컬레이션 / 종료 판정
        for code in list(self.active.keys()):
            incident = self.active[code]

            # 이번 프레임에 감지되지 않았고, 유예 시간도 지났으면 상황 종료
            if code not in seen_codes and (now - incident.last_seen) >= self._clear_sec(code):
                del self.active[code]
                self._ended_at[code] = now
                continue

            # 감지가 계속 들어와도 최대 지속 시간을 넘기면 강제 종료한다.
            # 이게 없으면 상황이 영구히 진행중으로 갇혀서 새 사건을 전부 묵살한다(위 주석 참고).
            if incident.duration(now) >= self._max_sec(code):
                print("[incident] code=%s 상황이 %.0f분 넘게 이어져 강제 종료 - 다음 감지는 새 건으로 처리"
                      % (code, incident.duration(now) / 60.0))
                del self.active[code]
                self._ended_at[code] = now
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
    def status_line(self, now=None):
        """디버그 화면/로그에 현재 진행 중인 상황을 한 줄로 표시.

        now를 반드시 넘길 것: 영상 파일로 돌릴 때는 시간 기준이 '영상 내부 타임스탬프'라서,
        여기서 time.time()을 쓰면 지속시간이 엉뚱하게 표시된다(실제로 그런 버그가 있었다).
        """
        if not self.active:
            return ""
        now = now if now is not None else time.time()
        parts = []
        for code, inc in sorted(self.active.items()):
            # 남은 최대 지속 시간을 같이 보여줘서, 왜 새 이벤트가 안 뜨는지 바로 알 수 있게 한다
            left = max(0.0, self._max_sec(code) - inc.duration(now))
            # 흡수된 감지 횟수를 같이 보여주면 "탐지는 되는데 전송만 막힌 상태"가 한눈에 보인다
            extra = " 흡수%d회" % inc.absorbed if inc.absorbed else ""
            parts.append("%s(%.0fs, 해제까지 %.0fs%s)" % (code, inc.duration(now), left, extra))
        return "진행중: " + " ".join(parts)
