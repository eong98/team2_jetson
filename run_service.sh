#!/bin/bash
# run_service.sh - 그냥 돌리기용 (화면 없음, 백그라운드 상주)
#   사용법:  ./run_service.sh start   <- 백그라운드 실행
#            ./run_service.sh stop    <- 종료
#            ./run_service.sh log     <- 로그 실시간 보기
#            ./run_service.sh status  <- 동작 확인
cd "$(dirname "$0")"
PIDFILE=worker.pid
LOGFILE=worker.log

case "$1" in
  start)
    if [ -f "$PIDFILE" ] && kill -0 "$(cat $PIDFILE)" 2>/dev/null; then
        echo "이미 실행 중입니다 (PID $(cat $PIDFILE))"; exit 1
    fi
    # nohup: 터미널(SSH)이 끊겨도 프로세스가 살아있게 함
    nohup python3 -u worker_skeleton.py --no-display >> "$LOGFILE" 2>&1 &
    echo $! > "$PIDFILE"
    echo "시작됨 (PID $!). 로그: ./run_service.sh log"
    ;;
  stop)
    if [ -f "$PIDFILE" ]; then
        kill "$(cat $PIDFILE)" 2>/dev/null && echo "종료됨"
        rm -f "$PIDFILE"
    else
        echo "실행 중이 아닙니다"
    fi
    ;;
  log)    tail -f "$LOGFILE" ;;
  status)
    if [ -f "$PIDFILE" ] && kill -0 "$(cat $PIDFILE)" 2>/dev/null; then
        echo "실행 중 (PID $(cat $PIDFILE))"
    else
        echo "중지 상태"
    fi
    ;;
  *) echo "사용법: $0 {start|stop|log|status}" ;;
esac
