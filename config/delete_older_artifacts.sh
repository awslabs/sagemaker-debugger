KEEP_FOR_DAYS=180

#RUN ONLY FOR NIGHTLY PIPELINE - variable is set in CODEPIPELINE JOB
if [ ! -z "${SMDEBUG_NIGHTLY_LATEST}" ]; 
then
BUCKET_NAME=tornasole-smdebug-nightly-binaries

DELETE_BEFORE=$(date --date '-'"$KEEP_FOR_DAYS"' days' +%F)
LIST_OLDER=$(aws s3api list-objects-v2 --bucket "$BUCKET_NAME" --query 'Contents[?LastModified < `'"$DELETE_BEFORE"'`].Key')
LIST_OLDER=${LIST_OLDER:1:${#LIST_OLDER}-2}
LIST_OLDER=(${LIST_OLDER//,/ });

for old_artifact in ${LIST_OLDER[@]}; 
do 
    stripped_old_artifact=${old_artifact:1:${#old_artifact[@]}-2}
    aws s3 rm "s3://${BUCKET_NAME}/${stripped_old_artifact}"; 
done

echo "Deleted all artifacts before $DELETE_BEFORE";
fi