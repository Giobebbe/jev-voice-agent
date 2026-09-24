#!/bin/zsh
# Run a command inside Terminal.app (which holds the mic, camera and screen permissions),
# log its output and write a DONE marker with the exit code.
#   scripts/in_terminal.sh python evals/live_eval.py
# Prints the log path; poll it for "__DONE__ <code>".
set -e
PROJECT="${0:A:h:h}"
mkdir -p "$PROJECT/runs"
STAMP=$(date +%Y%m%d-%H%M%S)
LOG="$PROJECT/runs/terminal-$STAMP.log"
JOB="$PROJECT/runs/job-$STAMP.sh"
{
  echo "cd ${(q)PROJECT}"
  echo "source .venv/bin/activate"
  echo "export PYTHONUNBUFFERED=1"
  echo "${(j: :)${(q)@}} > ${(q)LOG} 2>&1"
  echo "echo __DONE__ \$? >> ${(q)LOG}"
} > "$JOB"
osascript -e 'on run argv' -e 'tell application "Terminal" to do script "zsh " & quoted form of item 1 of argv' \
  -e 'end run' "$JOB" > /dev/null
echo "$LOG"
