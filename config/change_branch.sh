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

cd $CODEBUILD_SRC_DIR_RULES
export CODEBUILD_GIT_BRANCH_RULES="$(git symbolic-ref HEAD --short 2>/dev/null)"
if [ "$CODEBUILD_SRC_DIR_RULES" = "" ] ; then
  CODEBUILD_GIT_BRANCH_RULES="$(git branch -a --contains HEAD | sed -n 2p | awk '{ printf $1 }')";
  export CODEBUILD_GIT_BRANCH_RULES=${CODEBUILD_GIT_BRANCH_RULES#remotes/origin/};
fi

cd $CODEBUILD_SRC_DIR_RULES && git checkout $CODEBUILD_GIT_BRANCH_RULES
export RULES_CODEBUILD_SRC_DIR="$CODEBUILD_SRC_DIR_RULES"
export CURRENT_COMMIT_HASH_RULES=$(git log -1 --pretty=%h);
export CURRENT_REPO_NAME_RULES=$(basename `git rev-parse --show-toplevel`) ;
cd ..


export CODEBUILD_ACCOUNT_ID=$(aws sts get-caller-identity --query 'Account' --output text)
export CODEBUILD_PROJECT=${CODEBUILD_BUILD_ID%:$CODEBUILD_LOG_PATH}

export CODEBUILD_BUILD_URL=https://$AWS_DEFAULT_REGION.console.aws.amazon.com/codebuild/home?region=$AWS_DEFAULT_REGION#/builds/$CODEBUILD_BUILD_ID/view/new

echo "INFO =============================BUILD STARTED==================================="
echo "INFO =============================Build details========================== ::"
echo "INFO CODEBUILD_CURRENT_BUILD_URL = $CODEBUILD_BUILD_URL"
#echo "INFO CURRENT_REPO_NAME = $CURRENT_REPO_NAME"
#echo "INFO CURRENT_COMMIT_DATE = $CURRENT_COMMIT_DATE"
echo "INFO CODEBUILD_ACCOUNT_ID = $CODEBUILD_ACCOUNT_ID"
echo "INFO CURRENT_GIT_BRANCH = $CODEBUILD_GIT_BRANCH"
echo "INFO CURRENT_GIT_COMMIT = $CURRENT_COMMIT_HASH"
echo "INFO CODEBUILD_PROJECT = $CODEBUILD_PROJECT"
