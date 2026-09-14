#!/bin/bash
# run_watch.sh - 화면 보면서 테스트할 때 (튜닝용)
#   사용법:  ./run_watch.sh                  <- 카메라
#            ./run_watch.sh test_02_damage.mp4 <- 테스트 영상
#   화면:    http://localhost:8090/  또는 내 PC에서 http://10.100.0.164:8090/
cd "$(dirname "$0")"
if [ -z "$1" ]; then
    python3 worker_skeleton.py --loiter 20
else
    python3 worker_skeleton.py --source "$1" --loop --loiter 20
fi
