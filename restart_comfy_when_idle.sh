#!/bin/bash
PORT=6006
PID=952
LOG=/root/ComfyUI/comfyui.log
EMPTY_HITS=0
echo "$(date '+%F %T') watcher started: waiting for queue to drain before restart"
while true; do
  Q=$(curl -s http://127.0.0.1:$PORT/queue | python3 -c "import json,sys; d=json.load(sys.stdin); print(len(d.get('queue_running',[]))+len(d.get('queue_pending',[])))" 2>/dev/null)
  if [ -z "$Q" ]; then
    echo "$(date '+%F %T') ERROR: queue query failed (server down?) - aborting, NOT restarting"
    exit 1
  fi
  if [ "$Q" = "0" ]; then
    EMPTY_HITS=$((EMPTY_HITS+1))
    echo "$(date '+%F %T') queue empty (check $EMPTY_HITS/2)"
    if [ $EMPTY_HITS -ge 2 ]; then break; fi
  else
    EMPTY_HITS=0
    echo "$(date '+%F %T') still busy: $Q task(s)"
  fi
  sleep 30
done
echo "$(date '+%F %T') queue drained, restarting ComfyUI..."
sleep 5
if ps -p $PID -o cmd= 2>/dev/null | grep -q "main.py"; then
  kill $PID
  for i in $(seq 1 60); do kill -0 $PID 2>/dev/null || break; sleep 1; done
  kill -9 $PID 2>/dev/null
  echo "$(date '+%F %T') old process $PID stopped"
else
  echo "$(date '+%F %T') pid $PID not ComfyUI anymore, killing by pattern"
  pkill -f "main.py --listen 0.0.0.0.*--port $PORT" || true
  sleep 5
fi
cd /root/ComfyUI
nohup /root/miniconda3/bin/python main.py --listen 0.0.0.0 --disable-auto-launch --enable-assets --cache-none --novram --port $PORT >> $LOG 2>&1 &
NEWPID=$!
echo "$(date '+%F %T') new ComfyUI started pid=$NEWPID, waiting for http..."
for i in $(seq 1 120); do
  if curl -s -o /dev/null http://127.0.0.1:$PORT/; then
    echo "$(date '+%F %T') SUCCESS: ComfyUI is UP (pid $NEWPID, port $PORT)"
    exit 0
  fi
  sleep 5
done
echo "$(date '+%F %T') WARN: http not up after 600s, last log lines:"
tail -20 $LOG
exit 1
