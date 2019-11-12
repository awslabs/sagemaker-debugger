# Standard Library
import argparse
import os
import subprocess
import sys

parser = argparse.ArgumentParser(description="Build Tornasole binaries")
parser.add_argument(
    "--upload",
    default=False,
    dest="upload",
    action="store_true",
    help="Pass --upload if you want to upload the binaries" "built to the s3 location",
)
parser.add_argument(
    "--replace-latest",
    default=False,
    dest="replace_latest",
    action="store_true",
    help="Pass --replace-latest if you want to upload the new binary to "
    "replace the latest binary in the S3 location. Note that"
    "this also requires you to pass --upload",
)
args = parser.parse_args()
exec(open("smdebug/_version.py").read())

VERSION = __version__
BINARIES = ["mxnet", "tensorflow", "pytorch", "xgboost", "rules"]

for b in BINARIES:
    if b == "rules":
        env_var = "TORNASOLE_FOR_RULES"
    else:
        env_var = "TORNASOLE_WITH_" + b.upper()
    env = dict(os.environ)
    env[env_var] = "1"
    subprocess.check_call([sys.executable, "setup.py", "bdist_wheel", "--universal"], env=env)
    if args.upload:
        subprocess.check_call(
            [
                "aws",
                "s3",
                "cp",
                "dist/tornasole-{}-py2.py3-none-any.whl".format(VERSION),
                "s3://tornasole-binaries-use1/tornasole_{}/py3/".format(b),
            ]
        )

        if args.replace_latest:
            # upload current version
            subprocess.check_call(
                [
                    "aws",
                    "s3",
                    "cp",
                    "s3://tornasole-binaries-use1/tornasole_{}/py3/tornasole-{}-py2.py3-none-any.whl".format(
                        b, VERSION
                    ),
                    "s3://tornasole-binaries-use1/tornasole_{}/py3/latest/".format(b),
                ]
            )
            # remove other versions
            subprocess.check_call(
                [
                    "aws",
                    "s3",
                    "rm",
                    "--recursive",
                    "--exclude",
                    "tornasole-{}*".format(VERSION),
                    "s3://tornasole-binaries-use1/tornasole_{}/py3/latest/".format(b),
                ]
            )
    subprocess.check_call(["rm", "-rf", "dist", "build", "*.egg-info", ".eggs"])
