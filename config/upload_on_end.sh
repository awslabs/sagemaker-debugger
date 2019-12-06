cat $CODEBUILD_SRC_DIR/upload/$CURRENT_COMMIT_PATH/pytest_reports/*.html >> $CODEBUILD_SRC_DIR/upload/$CURRENT_COMMIT_PATH/pytest_reports/all_tests.html


upload_dirs() {
for var in "$@"
do
    aws s3 sync --quiet $CODEBUILD_SRC_DIR/upload/$CURRENT_COMMIT_PATH/$var s3://smdebugcodebuildtest/$CURRENT_COMMIT_PATH/$var
done
}

del_dirs() {
for var in "$@"
do
    aws s3 rm --recursive --quiet s3://smdebugcodebuildtest/$CURRENT_COMMIT_PATH/$var
done
}

PR_ID=$(echo $CODEBUILD_WEBHOOK_TRIGGER | cut -d '/' -f 2-)
export GITHUB_PR_URL=https://github.com/awslabs/$CURRENT_REPO_NAME/pull/$PR_ID
export S3_TEST_REPORT_URL=https://s3.console.aws.amazon.com/s3/object/smdebugcodebuildtest/$CURRENT_COMMIT_PATH/pytest_reports/all_tests.html?region=us-west-1

if [ $CODEBUILD_BUILD_SUCCEEDING -eq 0 ]
then
    upload_dirs local_trials integration_tests_logs pytest_reports
    echo "ERROR BUILD FAILED , ACCESS BUILD LOGS THROUGH GITHUB OR TROUGH THE LINK PR: $GITHUB_PR_URL . CODEBUILD: $CODEBUILD_BUILD_URL . Test logs are on S3 here: $S3_TEST_REPORT_URL"
else
    del_dirs s3_trials
    upload_dirs integration_tests_logs pytest_reports wheels
    echo "INFO BUILD SUCCEEDED!!! , ACCESS BUILD LOGS THROUGH GITHUB OR TROUGH THE LINK PR: $GITHUB_PR_URL . CODEBUILD: $CODEBUILD_BUILD_URL . Test logs are on S3 here: $S3_TEST_REPORT_URL"
fi
