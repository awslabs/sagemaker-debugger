#!/usr/bin/env python
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
import os
import sys
from datetime import date
import shutil
import logging
import urllib
import urllib.request
import tempfile
from zipfile import ZipFile
from subprocess import check_call

# Third Party
import setuptools

# First Party
import smdebug


DOCLINES = (__doc__ or "").split("\n")
FRAMEWORKS = ["tensorflow", "pytorch", "mxnet", "xgboost"]
TESTS_PACKAGES = ["pytest", "torchvision", "pandas"]
INSTALL_REQUIRES = ["protobuf>=3.6.0", "numpy", "packaging", "boto3>=1.10.32"]


def _protoc_bundle():
    import platform
    system = platform.system()
    machine = platform.machine()
    if system == 'Darwin':
        archive_url = 'https://github.com/protocolbuffers/protobuf/releases/download/v3.11.4/protoc-3.11.4-osx-x86_64.zip'
    elif system == 'Linux':
        archive_url = f'https://github.com/protocolbuffers/protobuf/releases/download/v3.11.4/protoc-3.11.4-linux-{machine}.zip'
    else:
        archive_url = None
    return archive_url


def get_protoc():
    """make sure protoc is available, otherwise download it and return the path to protoc"""
    if not shutil.which('protoc'):
        archive_url = _protoc_bundle()
        if not archive_url:
            raise RuntimeError("protoc not installed and I don't know how to download it, please install manually.")
        logging.info("Downloading protoc")
        (fname, headers) = urllib.request.urlretrieve(archive_url)
        tmpdir = tempfile.mkdtemp(prefix='protoc_smdebug')
        with ZipFile(fname, 'r') as zipf:
            zipf.extractall(tmpdir)
        protoc_bin = os.path.join(tmpdir, 'bin', 'protoc')
        os.chmod(protoc_bin, 0o755)
        return protoc_bin
    return shutil.which('protoc')



def compile_protobuf():
    proto_paths = ["smdebug/core/tfevent/proto"]
    protoc_bin = get_protoc()
    for proto_path in proto_paths:
        proto_files = os.path.join(proto_path, "*.proto")
        print("compiling protobuf files in {}".format(proto_path))
    check_call([protoc_bin, proto_path, "--python_out=."])


def build_package(version):
    compile_protobuf()
    packages = setuptools.find_packages(include=["smdebug", "smdebug.*"])
    setuptools.setup(
        name="smdebug",
        version=version,
        long_description="\n".join(DOCLINES[1:]),
        long_description_content_type="text/x-rst",
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
        return smdebug.__version__.strip()

    return smdebug.__version__.strip() + "b" + str(date.today()).replace("-", "")


version = detect_smdebug_version()
build_package(version=version)
