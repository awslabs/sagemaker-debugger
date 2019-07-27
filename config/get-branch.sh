#$CODEBUILD_WEBHOOK_BASE_REF IS DESTINATION BRANCH FOR PR.
#$CODEBUILD_GIT_BRANCH IS CURRENT BRANCH FOR THE REPO WHICH TRIGGERED BUILD.
core_repo="tornasole_core"
rules_repo="tornasole_rules"
tf_repo="tornasole_tf"
mxnet_repo="tornasole_mxnet"

export CODEBUILD_GIT_BRANCH="$(git symbolic-ref HEAD --short 2>/dev/null)"
if [ "$CODEBUILD_GIT_BRANCH" = "" ] ; then
  CODEBUILD_GIT_BRANCH="$(git branch -a --contains HEAD | sed -n 2p | awk '{ printf $1 }')";
  export CODEBUILD_GIT_BRANCH=${CODEBUILD_GIT_BRANCH#remotes/origin/};
fi
SUBSTRING=$(echo $CODEBUILD_WEBHOOK_BASE_REF| cut -d'/' -f 3)
BRANCH=''
if  [ "$CODEBUILD_WEBHOOK_EVENT" = "PULL_REQUEST_CREATED" ] || [ "$CODEBUILD_WEBHOOK_EVENT" = "PULL_REQUEST_REOPENED" ] || [ "$CODEBUILD_WEBHOOK_EVENT" = "PULL_REQUEST_UPDATED" ]; then
      BRANCH=$SUBSTRING

elif [ "$CODEBUILD_WEBHOOK_EVENT" != "PULL_REQUEST_CREATED" ] && [ "$CODEBUILD_WEBHOOK_EVENT" != "PULL_REQUEST_REOPENED" ] && [ "$CODEBUILD_WEBHOOK_EVENT" != "PULL_REQUEST_UPDATED" ] && [ "$CODEBUILD_GIT_BRANCH" != "alpha" ] && [ "$CODEBUILD_GIT_BRANCH" != "master" ] ; then
     if [ $(git merge-base --is-ancestor $CODEBUILD_GIT_BRANCH  "alpha" ; echo $?) -eq 1 ]; then
          BRANCH='alpha'

     elif [ $(git merge-base --is-ancestor $CODEBUILD_GIT_BRANCH  "alpha" ; echo $?) -eq 0 ]; then
          BRANCH='master'
    
     fi    
     
else BRANCH=$CODEBUILD_GIT_BRANCH
fi

TF_BRANCH=$BRANCH ;
CORE_BRANCH=$BRANCH ;
RULES_BRANCH=$BRANCH ;
MXNET_BRANCH=$BRANCH  ;


if [ "$CODEBUILD_GIT_BRANCH" != "alpha" ] && [ "$CODEBUILD_GIT_BRANCH" != "master" ] ; then
    file="configure_branch_for_test.txt"
    while IFS=: read -r repo_name default_or_branchname
    do
                if [ "$repo_name" = "$tf_repo" ] && [ "$default_or_branchname" != "default" ]; then
                        TF_BRANCH=$default_or_branchname
                elif [ "$repo_name" = "$mxnet_repo" ] && [ "$default_or_branchname" != "default" ] ; then
                        MXNET_BRANCH=$default_or_branchname
                elif [ "$repo_name" = "$rules_repo" ] && [ "$default_or_branchname" != "default" ] ; then
                        RULES_BRANCH=$default_or_branchname
                elif [ "$repo_name" = "$core_repo" ] && [ "$default_or_branchname" != "default" ] ; then
                        CORE_BRANCH=$default_or_branchname
                fi

    done <"$file"
fi

cd $CODEBUILD_SRC_DIR && git checkout $CODEBUILD_GIT_BRANCH
export CURRENT_COMMIT_HASH=$(git log -1 --pretty=%h);
export CURRENT_COMMIT_DATE="$(git show -s --format=%ci | cut -d' ' -f 1)_$(git show -s --format=%ci | cut -d' ' -f 2)"; 
export CURRENT_REPO_NAME=$(basename `git rev-parse --show-toplevel`) ;
export CURRENT_COMMIT_PATH="$CODEBUILD_SRC_DIR/wheels/$CURRENT_COMMIT_DATE/$CURRENT_REPO_NAME/$CURRENT_COMMIT_HASH"

if  [ "$CURRENT_REPO_NAME" != "$core_repo" ]; then
    cd $CODEBUILD_SRC_DIR_tornasole_core && git checkout $CORE_BRANCH 
    export CORE_REPO_NAME=$(basename `git rev-parse --show-toplevel`) ;
    export CORE_COMMIT_HASH=$(git log -1 --pretty=%h);
    export CORE_COMMIT_DATE="$(git show -s --format=%ci | cut -d' ' -f 1)_$(git show -s --format=%ci | cut -d' ' -f 2)"; 
    export CORE_PATH="$CODEBUILD_SRC_DIR/wheels/$CORE_COMMIT_DATE/$CORE_REPO_NAME/$CORE_COMMIT_HASH"
    cd ..
fi

if  [ "$CURRENT_REPO_NAME" != "$rules_repo"  ]; then
    cd $CODEBUILD_SRC_DIR_tornasole_rules && git checkout $RULES_BRANCH 
    export RULES_REPO_NAME=$(basename `git rev-parse --show-toplevel`) ;
    export RULES_COMMIT_HASH=$(git log -1 --pretty=%h);
    export RULES_COMMIT_DATE="$(git show -s --format=%ci | cut -d' ' -f 1)_$(git show -s --format=%ci | cut -d' ' -f 2)"; 
    export RULES_PATH="$CODEBUILD_SRC_DIR/wheels/$RULES_COMMIT_DATE/$RULES_REPO_NAME/$RULES_COMMIT_HASH"
    cd ..
fi

if  [ "$CURRENT_REPO_NAME" != "$mxnet_repo" ]; then
    cd $CODEBUILD_SRC_DIR_tornasole_mxnet && git checkout $MXNET_BRANCH 
    export MXNET_REPO_NAME=$(basename `git rev-parse --show-toplevel`) ;
    export MXNET_COMMIT_HASH=$(git log -1 --pretty=%h);
    export MXNET_COMMIT_DATE="$(git show -s --format=%ci | cut -d' ' -f 1)_$(git show -s --format=%ci | cut -d' ' -f 2)"; 
    export MXNET_PATH="$CODEBUILD_SRC_DIR/wheels/$MXNET_COMMIT_DATE/$MXNET_REPO_NAME/$MXNET_COMMIT_HASH"
    cd ..
fi

if  [ "$CURRENT_REPO_NAME" != "$tf_repo" ]; then
    cd $CODEBUILD_SRC_DIR_tornasole_tf && git checkout $TF_BRANCH 
    export TF_REPO_NAME=$(basename `git rev-parse --show-toplevel`) ;
    export TF_COMMIT_HASH=$(git log -1 --pretty=%h);
    export TF_COMMIT_DATE="$(git show -s --format=%ci | cut -d' ' -f 1)_$(git show -s --format=%ci | cut -d' ' -f 2)"; 
    export TF_PATH="$CODEBUILD_SRC_DIR/wheels/$TF_COMMIT_DATE/$TF_REPO_NAME/$TF_COMMIT_HASH"
    cd ..
fi


            
export TF_BRANCH ;
export CORE_BRANCH ;
export RULES_BRANCH ;
export MXNET_BRANCH ;


