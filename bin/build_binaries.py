# Standard Library
import argparse
import os
import subprocess
import sys

parser = argparse.ArgumentParser(description="Build smdebug binaries")
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
parser.add_argument("--s3-prefix", default="s3://tornasole-bugbash-1113/binaries")
args = parser.parse_args()
exec(open("smdebug/_version.py").read())

VERSION = __version__

subprocess.check_call([sys.executable, "setup.py", "bdist_wheel"])
path = "smdebug"
if args.upload:
    subprocess.check_call(
        [
            "aws",
            "s3",
            "cp",
            "dist/smdebug-{}-py3-none-any.whl".format(VERSION),
            os.path.join(args.s3_prefix, path, ""),
        ]
    )

    if args.replace_latest:
        # upload current version
        subprocess.check_call(
            [
                "aws",
                "s3",
                "cp",
                os.path.join(args.s3_prefix, path, f"smdebug-{VERSION}-py3-none-any.whl"),
                os.path.join(args.s3_prefix, path, "latest", ""),
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
                f"smdebug-{VERSION}*",
                os.path.join(args.s3_prefix, path, "latest"),
            ]
        )
subprocess.check_call(["rm", "-rf", "dist", "build", "*.egg-info", ".eggs"])
