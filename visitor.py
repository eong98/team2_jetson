# -*- coding: utf-8 -*-
"""
visitor.py

CCTV_VISITOR 테이블에 넣을 손님(방문객) 입·퇴장을 감지한다.

CCTV_VISITOR 컬럼: NO / CNO / TRACK_ID / INTIME / OUTTIME / STAYTIME / STATE / CDATE
  - TRACK_ID : AI가 부여한 추적 ID (문자열)
  - INTIME   : 입장 시각 'YYYY-MM-DD HH:MM:SS'
  - OUTTIME  : 퇴장 시각 (입장 중이면 NULL)
  - STAYTIME : 체류 시간(분). 화면에서 "N분"으로 표시됨
  - STATE    : 0=입장중, 1=정상퇴장, 2=장시간체류

[동작]
  사람이 새로 잡힘  -> 바로 입장 처리하지 않고 후보로 둔다
  MIN_VISIT_SEC 이상 계속 보임 -> 입장 확정, 서버에 INSERT 요청 (STATE=0)
  EXIT_SEC 이상 안 보임        -> 퇴장 확정, 서버에 UPDATE 요청 (OUTTIME/STAYTIME/STATE)

[후보 단계를 두는 이유]
사람이 한두 프레임 잘못 잡히거나(오탐), 화면 가장자리를 스쳐 지나가기만 해도 방문객으로
집계되면 통계가 엉망이 된다. 몇 초 이상 실제로 머문 경우만 손님으로 인정한다.

트래커가 짧은 미검출(1.5초)은 이미 흡수해주므로, 여기서 보는 '안 보임'은 그보다 긴 경우다.
"""

import time

MIN_VISIT_SEC = 3.0        # 이 시간 이상 보여야 손님으로 인정 (스쳐 지나감/오탐 제외)
EXIT_SEC = 5.0             # 이 시간 이상 안 보이면 퇴장으로 확정
LONG_STAY_SEC = 300.0      # 이 이상 머물면 STATE=2(장시간체류)로 기록 (기본 5분)

STATE_IN = 0               # 입장중
STATE_OUT = 1              # 정상퇴장
STATE_LONG = 2             # 장시간체류


def _fmt(ts):
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(ts))


class Visit(object):
    def __init__(self, cno, track_id, first_seen):
        self.track_id = track_id
        self.first_seen = first_seen
        self.last_seen = first_seen
        self.entered = False       # 입장 확정(서버 전송 완료) 여부
        # DB의 TRACK_ID로 쓸 문자열. 워커를 재시작하면 track_id가 1부터 다시 시작하므로
        # 입장 시각을 붙여서 다른 날/다른 실행과 겹치지 않게 만든다.
        self.visit_id = "%s-C%d-T%d" % (
            time.strftime("%Y%m%d%H%M%S", time.localtime(first_seen)), cno, track_id)

    def stay_sec(self):
        return max(0.0, self.last_seen - self.first_seen)


class VisitorTracker(object):
    """트래커 결과를 받아 입장/퇴장 이벤트를 만들어준다."""

    def __init__(self, cno, min_visit_sec=MIN_VISIT_SEC, exit_sec=EXIT_SEC,
                 long_stay_sec=LONG_STAY_SEC):
        self.cno = cno
        self.min_visit_sec = min_visit_sec
        self.exit_sec = exit_sec
        self.long_stay_sec = long_stay_sec
        self.visits = {}           # track_id -> Visit

    # -----------------------------------------------------------------
    def update(self, tracked, now=None):
        """tracked: [(track_id, box), ...]

        반환: (입장 목록, 퇴장 목록) - 각각 서버로 보낼 dict 리스트.
        사람이 안 잡히는 프레임에도 호출해야 퇴장 판정이 된다.
        """
        now = now if now is not None else time.time()
        enters, exits = [], []

        seen = set()
        for track_id, _box in tracked:
            seen.add(track_id)
            visit = self.visits.get(track_id)
            if visit is None:
                visit = Visit(self.cno, track_id, now)
                self.visits[track_id] = visit
            else:
                visit.last_seen = now

            # 충분히 오래 보였으면 입장 확정
            if not visit.entered and visit.stay_sec() >= self.min_visit_sec:
                visit.entered = True
                enters.append({
                    "cno": self.cno,
                    "trackId": visit.visit_id,
                    "intime": _fmt(visit.first_seen),
                })

        # 안 보이는 트랙들 퇴장 판정
        for track_id in list(self.visits.keys()):
            if track_id in seen:
                continue
            visit = self.visits[track_id]
            if (now - visit.last_seen) < self.exit_sec:
                continue

            if visit.entered:
                exits.append(self._make_exit(visit))
            # 입장 확정 전에 사라진 건 손님으로 세지 않고 그냥 버린다
            del self.visits[track_id]

        return enters, exits

    # -----------------------------------------------------------------
    def _make_exit(self, visit):
        stay = visit.stay_sec()
        return {
            "cno": self.cno,
            "trackId": visit.visit_id,
            "outtime": _fmt(visit.last_seen),
            # STAYTIME은 '분' 단위 컬럼. 1분 미만은 0분으로 기록된다.
            "staytime": int(round(stay / 60.0)),
            "state": STATE_LONG if stay >= self.long_stay_sec else STATE_OUT,
        }

    def flush(self, now=None):
        """워커를 정상 종료할 때, 아직 입장 중인 손님들을 퇴장 처리한다.
        이걸 안 하면 STATE=0(입장중)인 행이 영원히 남는다."""
        now = now if now is not None else time.time()
        exits = []
        for track_id in list(self.visits.keys()):
            visit = self.visits[track_id]
            if visit.entered:
                visit.last_seen = min(visit.last_seen, now)
                exits.append(self._make_exit(visit))
            del self.visits[track_id]
        return exits

    @property
    def active_count(self):
        """현재 매장 안에 있는 것으로 집계된 손님 수 (디버그 표시용)."""
        return sum(1 for v in self.visits.values() if v.entered)
