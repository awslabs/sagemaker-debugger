if [ -z "$framework" ]
  then
    echo "framework is not mentioned"
    exit 1
fi

if [ "$framework" = "tensorflow" ]
  then
    echo "Launching testing job using $framework framework"


elif [ "$framework" = "mxnet" ]
  then
    echo "Launching testing job using $framework framework"

else
    echo "$framework framework not supported!!!"
    exit 1

fi

export TORNASOLE_LOG_LEVEL=debug
python -m pytest tests/
