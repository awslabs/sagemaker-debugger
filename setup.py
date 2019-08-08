import os
import sys
import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

def compile_summary_protobuf():
    proto_path = 'tornasole_core/tfevent'
    proto_files = os.path.join(proto_path, '*.proto')
    cmd = 'protoc ' + proto_files + ' --python_out=.'
    print('compiling protobuf files in {}'.format(proto_path))
    return os.system('set -ex &&' + cmd)


if compile_summary_protobuf() != 0:
    print('ERROR: Compiling summary protocol buffers failed. You will not be '
          'able to use the logging APIs for visualizing MXNet data in TensorBoard. '
          'Please make sure that you have installed protobuf3 compiler and runtime correctly.')
    sys.exit(1)

setuptools.setup(
    name="tornasole_core",
    version="0.2",
    author="The Tornasole Team",
    author_email="tornasole@amazon.com",
    description="Tornasole Core",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/awslabs/tornasole_core",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
    #pinning aioboto3 version as aiobot3 is pinning versions https://github.com/aio-libs/aiobotocore/issues/718
    install_requires = ['aioboto3==6.4.1', 'nest_asyncio', 'protobuf>=3.6.0' ,'botocore==1.12.91','boto3==1.9.91', 'aiobotocore==0.10.2'],
    setup_requires=["pytest-runner"],
    tests_require=["pytest", "tensorflow"],
)
