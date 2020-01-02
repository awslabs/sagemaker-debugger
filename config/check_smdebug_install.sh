#!/bin/bash
set -ex
set -o pipefail
python -c "import smdebug.tf"
res="$?"
echo "output of import smdebug is: $res"
exit $res
