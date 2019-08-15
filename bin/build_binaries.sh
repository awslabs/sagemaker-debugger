export VERSION='0.3'

for FRAMEWORK in tensorflow mxnet pytorch
do
    CAPITALIZED_FRAMEWORK=`echo "$FRAMEWORK" | tr '[a-z]' '[A-Z]'`
    export TORNASOLE_WITH_$CAPITALIZED_FRAMEWORK=1
    python setup.py bdist_wheel --universal
    unset TORNASOLE_WITH_$CAPITALIZED_FRAMEWORK
#    aws s3 cp dist/tornasole-$VERSION-py2.py3-none-any.whl s3://tornasole-binaries-use1/tornasole_$FRAMEWORK/py3/
    rm -rf dist build *.egg-info
done
