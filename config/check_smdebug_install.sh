#!/bin/bash
set -ex
set -o pipefail
python -c "import smdebug"
res="$?"
echo "output of import smdebug is: $res"

if [ $1 ]; then
  python -c "import $1"
  res="$?"
  echo "output of import $1 is: $res"
  exit $res
fi
exit $res
