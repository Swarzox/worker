#!/bin/bash
set -e

N=${1:-6}
MODE=${2:-"vast"}  # "vast", "nofallback", or "fal"

echo "=== TEST: $N parallel requests ==="
rm -f /tmp/ztest_*.txt

FALLBACK_ARG=""

if [ "$MODE" = "nofallback" ]; then
  FALLBACK_ARG="--no-fallback"
  echo "(mode: Vast.ai, fallback DISABLED)"
elif [ "$MODE" = "fal" ]; then
  FALLBACK_ARG="--fal-only"
  echo "(mode: fal.ai only)"
else
  echo "(mode: Vast.ai with fal.ai fallback)"
fi

for i in $(seq 1 $N); do
  python client.py --prompt "Test image $i" $FALLBACK_ARG > /tmp/ztest_$i.txt 2>&1 &
done

echo "Waiting for all requests to complete..."
wait

echo ""
echo "=== Summary ==="
VAST_COUNT=$(grep -l "\[Vast.ai\]" /tmp/ztest_*.txt 2>/dev/null | xargs -I {} grep -l "Done in" {} 2>/dev/null | wc -l | tr -d ' ')
FAL_COUNT=$(grep -l "\[fal.ai\]" /tmp/ztest_*.txt 2>/dev/null | xargs -I {} grep -l "Done in" {} 2>/dev/null | wc -l | tr -d ' ')
FAILED_COUNT=$(grep -L "Done in" /tmp/ztest_*.txt 2>/dev/null | wc -l | tr -d ' ')

echo "Vast.ai: $VAST_COUNT"
echo "fal.ai:  $FAL_COUNT"
echo "Failed:  $FAILED_COUNT"

echo ""
echo "=== Worker Distribution ==="
grep -oh "→ http://[0-9.]*:[0-9]*" /tmp/ztest_*.txt 2>/dev/null | sed 's/→ //' | sort | uniq -c || echo "No workers"

echo ""
echo "=== Details ==="
for f in /tmp/ztest_*.txt; do
  echo "--- $(basename $f) ---"
  grep -E "→|Done|fal.ai|Error" "$f" | head -3 || echo "(no output)"
done
