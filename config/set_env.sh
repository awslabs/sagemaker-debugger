export CODEBUILD_GIT_BRANCH="$(git symbolic-ref HEAD --short 2>/dev/null)"
if [ "$CODEBUILD_GIT_BRANCH" = "" ] ; then
  CODEBUILD_GIT_BRANCH="$(git branch -a --contains HEAD | sed -n 2p | awk '{ printf $1 }')";
  export CODEBUILD_GIT_BRANCH=${CODEBUILD_GIT_BRANCH#remotes/origin/};
fi

cd $CODEBUILD_SRC_DIR && git checkout $CODEBUILD_GIT_BRANCH
export CURRENT_COMMIT_HASH=$(git log -1 --pretty=%h);
#export CURRENT_COMMIT_DATE="$(git show -s --format=%ci | cut -d' ' -f 1)$(git show -s --format=%ci | cut -d' ' -f 2)";
export CURRENT_DATETIME=$(date +'%Y%m%d_%H%M%S')
export CURRENT_REPO_NAME=$(basename `git rev-parse --show-toplevel`) ;
export CURRENT_COMMIT_PATH="$CURRENT_DATETIME/$CURRENT_COMMIT_HASH"
cd ..


export CODEBUILD_ACCOUNT_ID=$(aws sts get-caller-identity --query 'Account' --output text)
export CODEBUILD_PROJECT=${CODEBUILD_BUILD_ID%:$CODEBUILD_LOG_PATH}

export CODEBUILD_BUILD_URL=https://$AWS_DEFAULT_REGION.console.aws.amazon.com/codebuild/home?region=$AWS_DEFAULT_REGION#/builds/$CODEBUILD_BUILD_ID/view/new

echo "INFO =============================BUILD STARTED==================================="
echo "INFO =============================Build details========================== ::"
echo "INFO CODEBUILD_CURRENT_BUILD_URL = $CODEBUILD_BUILD_URL"
#echo "INFO CURRENT_REPO_NAME = $CURRENT_REPO_NAME"
echo "INFO CURRENT_COMMIT_DATE = $CURRENT_COMMIT_DATE"
echo "INFO CODEBUILD_ACCOUNT_ID = $CODEBUILD_ACCOUNT_ID"
echo "INFO CURRENT_GIT_BRANCH = $CODEBUILD_GIT_BRANCH"
echo "INFO CURRENT_GIT_COMMIT = $CODEBUILD_GIT_COMMIT"
echo "INFO CODEBUILD_PROJECT = $CODEBUILD_PROJECT"

PR_ID=$(echo $CODEBUILD_WEBHOOK_TRIGGER | cut -d ';' -f 2-)
export GITHUB_PR_URL=https://github.com/awslabs/$CURRENT_REPO_NAME/pull/$PR_ID
#https://s3.console.aws.amazon.com/s3/object/tornasolecodebuildtest/20190817_215022/c24a121/reports/all_tests.html?region=us-east-1&tab=overview
export S3_TEST_REPORT_URL=https://s3.console.aws.amazon.com/s3/object/tornasolecodebuildtest/$CURRENT_COMMIT_PATH/reports/all_tests.html
