#!/bin/bash
# run_watch.sh - 화면 보면서 테스트할 때 (튜닝용)
#
#   사용법:
#     bash run_watch.sh                              <- CSI 카메라
#     bash run_watch.sh fight.mp4                    <- 테스트 영상 (반복재생)
#     bash run_watch.sh fight.mp4 --codes 01         <- 폭행만 판정 (다른 코드 노이즈 제거)
#     bash run_watch.sh fight.mp4 --no-dedup         <- incident 억제 끄고 원본 발화 전부 보기
#     bash run_watch.sh fight.mp4 --no-report        <- 서버 전송 없이 판정만
#     bash run_watch.sh fight.mp4 --codes 01 --no-dedup --no-report   <- 여러 개 조합 가능
#
#   화면:  젯슨 VNC에서 http://localhost:8090/
#          다른 PC에서  http://10.100.0.164:8090/   (네트워크가 허용할 때만)
#
#   화면 읽는 법:
#     상단 assault x.x/2.0  = 폭행 누적 점수 게이지 (2.0 = 확정)
#          iou= gap= mot=   = 이번 프레임 실측값. 튜닝은 이 숫자를 보고 한다
#     하단 damage=% fire=% flicker=  = 02/06 지표
cd "$(dirname "$0")"

# [2026-09-18 추가] 파일을 하나씩 올리다 부분 갱신되면 카메라를 연 뒤에야
# 시그니처 불일치 오류가 터진다. 먼저 검증해서 낭비되는 대기 시간을 없앤다.
python3 check_files.py || {
    echo
    echo "파일 버전이 맞지 않아 실행을 중단했습니다. jetson_worker.zip을 다시 풀어주세요."
    exit 1
}
echo

# [2026-09-18 수정] 첫 번째 인자는 영상 경로, 그 뒤 인자들("${@:2}")은
# worker_skeleton.py로 그대로 넘긴다. 전에는 뒤에 붙인 --codes 같은 옵션이 무시됐다.
if [ -z "$1" ]; then
    python3 worker_skeleton.py --loiter 20
else
    python3 worker_skeleton.py --source "$1" --loop --loiter 20 "${@:2}"
fi
