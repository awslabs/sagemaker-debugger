#!/bin/bash

# This helper script returns a string "true" or "false" depending on whether the profiler integration tests should be
# run on this branch or not. This is determined by looking at the files changed from this branch to master.

# Get the branch we're running integration tests on.
export CODEBUILD_GIT_BRANCH="$(git symbolic-ref HEAD --short 2>/dev/null)"
if [ "$CODEBUILD_GIT_BRANCH" = "" ] ; then
  CODEBUILD_GIT_BRANCH="$(git branch -a --contains HEAD | sed -n 2p | awk '{ printf $1 }')";
  export CODEBUILD_GIT_BRANCH=${CODEBUILD_GIT_BRANCH#remotes/origin/};
fi

# To manually disable profiler integration tests from running in the PR CI, set this environment variable to "true".
# If you do this, remember to reset it back to "false" before merging the PR.
disable_integration_tests="true"
if [ $disable_integration_tests = "true" ]; then
  echo "false"
  exit 0
fi

# If the branch is master, then just run integration tests.
if [ $CODEBUILD_GIT_BRANCH = "master" ]; then
  echo "true"
  exit 0
fi

# Otherwise, check to see what files have changed in this branch. If files in smdebug/core or smdebug/profiler or
# smdebug/$framework have been modified, run the integration tests. Otherwise, don't run the integration tests.
for file in $(git diff --name-only master $CODEBUILD_GIT_BRANCH)
do
  folders=(${file//// })
  root_folder=${folders[0]}
  framework_folder=${folders[1]}
  if [ $root_folder = "smdebug" ] && [[ $framework_folder = "core" || $framework_folder = "profiler" || $framework_folder = "${1}" ]]; then
    echo "true"
    exit 0
  fi
done

echo "false"
