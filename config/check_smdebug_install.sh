#!/bin/bash
set -ex
set -o pipefail

echo "Checking if smdebug has been installed"
python -c "import smdebug"
res="$?"
if [ $res -gt 0 ]; then
  echo "output of import smdebug is: $res"
  exit $res
else
  echo "Successfully imported smdebug package."
fi

echo "Check if smdebug_rules has been installed"
python -c "import smdebug_rules"
res="$?"
if [ $res -gt 0 ]; then
  echo "output of import smdebug_rules is: $res"
  exit $res
else
  echo "Successfully imported smdebug_rules package"
fi

if [ $1 ]; then
  python -c "import $1"
  res="$?"
  echo "output of import $1 is: $res"
  exit $res
fi
exit $res
