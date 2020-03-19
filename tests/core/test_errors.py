# Standard Library
import os
import tempfile

# First Party
from smdebug.exceptions import InvalidCollectionConfiguration

# Local
from ..util import EnvManager


def test_exception_logging():
    with EnvManager("SM_TRAINING_ENV", "1"), tempfile.NamedTemporaryFile(
        mode="w+"
    ) as tmp, EnvManager("SMDEBUG_SM_LOGFILE", tmp.name):
        try:
            raise InvalidCollectionConfiguration("ra")
        except Exception as e:
            pass
        assert os.path.exists(tmp.name)
        tmp.flush()
        tmp.seek(0, 0)
        content = tmp.read()
        print("content: ")
        print(content)
        assert "InvalidCollectionConfiguration" in content
