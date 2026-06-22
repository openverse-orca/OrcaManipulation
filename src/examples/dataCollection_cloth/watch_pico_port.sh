#!/usr/bin/env bash
# 轮询 Pico TCP 8001：LISTEN / ESTAB / 对端 IP
set -u
DURATION="${1:-90}"
echo "[watch_pico] 监控 ${DURATION}s，检查 0.0.0.0:8001 ..."
for ((i=0; i<DURATION; i+=2)); do
  ts="$(date +%H:%M:%S)"
  listen="$(ss -tlnp 2>/dev/null | grep ':8001 ' || true)"
  estab="$(ss -tnp 2>/dev/null | grep ':8001 ' | grep -v LISTEN || true)"
  if [[ -n "$estab" ]]; then
    echo "[$ts] CONNECTED"
    echo "$estab"
  elif [[ -n "$listen" ]]; then
    echo "[$ts] LISTEN (等待 Pico 客户端连接...)"
  else
    echo "[$ts] NO_SERVER (8001 未监听，遥操进程可能未启动)"
  fi
  sleep 2
done
