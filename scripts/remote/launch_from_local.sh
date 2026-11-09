#!/usr/bin/env bash
# Local orchestrator v2: wait for the box, prepare it, send the new commits as a git bundle in
# 20 MB pieces (each retried and checksummed, so a dropped connection costs one piece), fast-forward
# the box's clone, then launch setup and every remaining run detached on the box.
set -uo pipefail
SP="$(cd "$(dirname "$0")" && pwd)"
PORT=${PORT:-10647}; HOST=${HOST:-ssh3.vast.ai}
OPTS="-o ConnectTimeout=15 -o BatchMode=yes -o StrictHostKeyChecking=no -o ServerAliveInterval=20 -o ServerAliveCountMax=6"
SSH="ssh $OPTS -p $PORT root@$HOST"
retry() { for t in 1 2 3 4 5 6; do "$@" && return 0; echo "  retry $t: $*" | cut -c1-120; sleep 10; done; return 1; }
for i in $(seq 1 240); do
  if timeout 25 $SSH true 2>/dev/null; then echo "connected after $i tries ($(date -u +%H:%M:%S) UTC)"; break; fi
  [ "$i" = 240 ] && { echo "GAVE UP: not reachable after 2 hours"; exit 1; }
  sleep 30
done
set -e
retry scp -q $OPTS -P "$PORT" "$SP/remote_prepare.sh" "$SP/remote_launch.sh" root@$HOST:/root/
retry $SSH 'bash /root/remote_prepare.sh' | grep -v -i "welcome\|have fun\|AI agents"
BOXHEAD=$($SSH 'git -C /root/mb5 rev-parse HEAD' 2>/dev/null | tail -1)
cd /c/Users/djtde/Downloads/ACL_1
LOCAL=$(git rev-parse HEAD)
if [ "$BOXHEAD" != "$LOCAL" ]; then
  rm -rf "$SP/bundle"; mkdir -p "$SP/bundle"
  git bundle create "$SP/bundle/mb5.bundle" "$BOXHEAD..main" 2>/dev/null
  (cd "$SP/bundle" && split -b 20m -d mb5.bundle part_ && sha256sum mb5.bundle > full.sha && sha256sum part_* > parts.sha)
  echo "bundle $(du -h "$SP/bundle/mb5.bundle" | cut -f1), $(ls "$SP/bundle"/part_* | wc -l) pieces"
  $SSH 'mkdir -p /root/bundle' >/dev/null 2>&1
  for p in "$SP/bundle"/part_*; do
    n=$(basename "$p"); want=$(grep " $n\$" "$SP/bundle/parts.sha" | cut -c1-64)
    have=$($SSH "sha256sum /root/bundle/$n 2>/dev/null | cut -c1-64" 2>/dev/null | tail -1)
    [ "$have" = "$want" ] && { echo "  $n already there"; continue; }
    retry scp -q $OPTS -P "$PORT" "$p" root@$HOST:/root/bundle/
    echo "  sent $n"
  done
  retry scp -q $OPTS -P "$PORT" "$SP/bundle/full.sha" root@$HOST:/root/bundle/
  retry $SSH 'cd /root/bundle && cat part_* > mb5.bundle && sha256sum -c full.sha && cd /root/mb5 && git pull -q --ff-only /root/bundle/mb5.bundle main && git log --oneline -1 && git status --short | head -3'
fi
T=$(cat ~/.cache/huggingface/token)
retry $SSH "HF_TOKEN=$T bash /root/remote_launch.sh" | grep -v -i "welcome\|have fun\|AI agents"
echo "LAUNCHED at $(date -u +%H:%M:%S) UTC"
