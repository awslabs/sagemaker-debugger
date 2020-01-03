#!/bin/bash
set -ex
set -o pipefail
python -c "import smdebug"
res="$?"
echo "output of import smdebug is: $res"

version=`python -c "exec(\"import smdebug\nprint(smdebug.__version__)\")`
# force check version, you can set this env variable in build env when releasing
if [ "$force_check_smdebug_version" != version ]; then
  echo "force_check_version $force_check_smdebug_version doesn't match version found: $version"
  exit 1
fi

if [ $1 ]; then
  python -c "import $1"
  res="$?"
  echo "output of import $1 is: $res"
  exit $res
fi
exit $res
