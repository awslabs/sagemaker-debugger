# Standard Library
import os
import sys

# Third Party
import setuptools

exec(open("smdebug/_version.py").read())
CURRENT_VERSION = __version__
FRAMEWORKS = ["tensorflow", "pytorch", "mxnet", "xgboost"]
TESTS_PACKAGES = ["pytest", "torchvision", "pandas"]
INSTALL_REQUIRES = [
    # aiboto3 implicitly depends on aiobotocore
    "aioboto3==6.4.1",  # no version deps
    "aiobotocore==0.10.4",  # pinned to a specific botocore & boto3
    "aiohttp>=3.6.0,<4.0",  # aiobotocore breaks with 4.0
    # boto3 explicitly depends on botocore
    "boto3==1.9.252",  # Sagemaker requires >= 1.9.213
    "botocore==1.12.252",
    "nest_asyncio",
    "protobuf>=3.6.0",
    "numpy",
    "packaging",
]


def compile_summary_protobuf():
    proto_paths = ["smdebug/core/tfevent/proto"]
    cmd = "set -ex && protoc "
    for proto_path in proto_paths:
        proto_files = os.path.join(proto_path, "*.proto")
        cmd += proto_files + " "
        print("compiling protobuf files in {}".format(proto_path))
    cmd += " --python_out=."
    return os.system(cmd)


def build_package(version):
    setuptools.setup(
        name="smdebug",
        version=version,
        # author="The Tornasole Team",
        # author_email="tornasole@amazon.com",
        description="Automated debugging for machine learning",
        # long_description=long_description,
        # long_description_content_type="text/markdown",
        url="https://github.com/awslabs/tornasole_core",
        packages=setuptools.find_packages(),
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
    )


if compile_summary_protobuf() != 0:
    print(
        "ERROR: Compiling summary protocol buffers failed. You will not be able to use smdebug. "
        "Please make sure that you have installed protobuf3 compiler and runtime correctly."
    )
    sys.exit(1)


def scan_git_secrets():
    import subprocess
    import os
    import shutil

    def git(*args):
        return subprocess.call(['git'] + list(args))

    shutil.rmtree('/tmp/git-secrets', ignore_errors=True)
    a = git("clone", "https://github.com/awslabs/git-secrets.git", "/tmp/git-secrets")
    dir_path = os.path.dirname(os.path.realpath(__file__))
    b = os.chdir("/tmp/git-secrets")
    c = subprocess.check_call(['make'] + ['install'])
    d = os.chdir(dir_path)
    e = git("secrets", "--install")
    f = git("secrets", "--register-aws")
    g = git("secrets", "--scan")
    return g


if scan_git_secrets() != 0:
    import sys

    sys.exit(1)

build_package(version=CURRENT_VERSION)
