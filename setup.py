#!/usr/bin/env python3
""" Amazon SageMaker Debugger is an offering from AWS which helps you automate the debugging of machine learning training jobs.
This library powers Amazon SageMaker Debugger, and helps you develop better, faster and cheaper models by catching common errors quickly.
It allows you to save tensors from training jobs and makes these tensors available for analysis, all through a flexible and powerful API.
It supports TensorFlow, PyTorch, MXNet, and XGBoost on Python 3.6+.
- Zero Script Change experience on SageMaker when using supported versions of SageMaker Framework containers or AWS Deep Learning containers
- Full visibility into any tensor which is part of the training process
- Real-time training job monitoring through Rules
- Automated anomaly detection and state assertions
- Interactive exploration of saved tensors
- Distributed training support
- TensorBoard support

"""

# Standard Library
import contextlib
import os
import sys
from datetime import date

# Third Party
import compile_protobuf
import setuptools

exec(open("smdebug/_version.py").read())
CURRENT_VERSION = __version__

DOCLINES = (__doc__ or "").split("\n")
FRAMEWORKS = ["tensorflow", "pytorch", "mxnet", "xgboost"]
TESTS_PACKAGES = ["pytest", "torchvision", "pandas"]
INSTALL_REQUIRES = [
    "protobuf>=3.6.0",
    "numpy>=1.16.0",
    "packaging",
    "boto3>=1.10.32",
    "pyinstrument>=3.1.3",
]


@contextlib.contextmanager
def remember_cwd():
    """
    Restore current directory when exiting context
    """
    curdir = os.getcwd()
    try:
        yield
    finally:
        os.chdir(curdir)


def scan_git_secrets():
    from subprocess import check_call
    import os
    import tempfile
    import shutil

    if os.path.exists(".git/hooks/pre-commit"):
        print("git secrets: pre-commit hook already present")
        return

    if shutil.which("git-secrets"):
        check_call(["git", "secrets", "--scan"])
        print("scanned for git secrets")

    else:
        with tempfile.TemporaryDirectory(prefix="git_secrets_") as tmpdir:
            check_call(["git", "clone", "https://github.com/awslabs/git-secrets.git", tmpdir])
            check_call([os.path.join(tmpdir, "git-secrets"), "--scan"])
        print("scanned for git secrets")


def build_package(version):
    compile_protobuf.compile_protobuf()
    packages = setuptools.find_packages(include=["smdebug", "smdebug.*"])
    setuptools.setup(
        name="smdebug",
        version=version,
        long_description="\n".join(DOCLINES[1:]),
        long_description_content_type="text/markdown",
        author="AWS DeepLearning Team",
        description=DOCLINES[0],
        url="https://github.com/awslabs/sagemaker-debugger",
        packages=packages,
        classifiers=[
            "Programming Language :: Python :: 3",
            "Programming Language :: Python :: 3.6",
            "Programming Language :: Python :: 3.7",
            "Programming Language :: Python :: 3.8",
            "Programming Language :: Python :: 3 :: Only",
            "License :: OSI Approved :: Apache Software License",
            "Operating System :: OS Independent",
        ],
        install_requires=INSTALL_REQUIRES,
        setup_requires=["pytest-runner"],
        tests_require=TESTS_PACKAGES,
        python_requires=">=3.6",
        license="Apache License Version 2.0",
    )


def scan_git_secrets():
    import subprocess
    import os
    import shutil

    def git(*args):
        return subprocess.call(["git"] + list(args))

    shutil.rmtree("/tmp/git-secrets", ignore_errors=True)
    git("clone", "https://github.com/awslabs/git-secrets.git", "/tmp/git-secrets")
    dir_path = os.path.dirname(os.path.realpath(__file__))
    os.chdir("/tmp/git-secrets")
    subprocess.check_call(["make"] + ["install"])
    os.chdir(dir_path)
    git("secrets", "--install")
    git("secrets", "--register-aws")
    return git("secrets", "--scan", "-r")


if scan_git_secrets() != 0:
    import sys

    sys.exit(1)


def detect_smdebug_version():
    if "--release" in sys.argv:
        sys.argv.remove("--release")
        return CURRENT_VERSION

    return CURRENT_VERSION + "b" + str(date.today()).replace("-", "")


version = detect_smdebug_version()
build_package(version=version)
