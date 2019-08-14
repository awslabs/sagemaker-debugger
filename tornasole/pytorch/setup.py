import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
	name="tornasole_pytorch",
    version="0.1",
    author="The Tornasole Team",
    author_email="tornasole@amazon.com",
    description="Tornasole Pytorch",
    long_description=long_description,
    long_description_content_type="text/markdown",
    # url="https://github.com/awslabs/tornasole_pytorch",
    install_requires=['tornasole_core', 'numpy'],
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
    setup_requires=["pytest-runner"],
    tests_require=["pytest", "numpy"]
)
