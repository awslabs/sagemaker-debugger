#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Standard Library
import glob
import logging
import os
import shutil
import sys
import tempfile
import urllib
import urllib.request
from subprocess import check_call
from zipfile import ZipFile


def script_name() -> str:
    """:returns: script name with leading paths removed"""
    return os.path.split(sys.argv[0])[1]


def config_logging():
    import time

    logging.getLogger().setLevel(logging.INFO)
    logging.basicConfig(format="{}: %(asctime)sZ %(levelname)s %(message)s".format(script_name()))
    logging.Formatter.converter = time.gmtime


def _protoc_bundle():
    """
    Returns an archive with the binary protoc distro for the platform
    """
    import platform

    system = platform.system()
    machine = platform.machine()
    if system == "Darwin":
        archive_url = "https://github.com/protocolbuffers/protobuf/releases/download/v3.11.4/protoc-3.11.4-osx-x86_64.zip"
        logging.info("Downloading protoc for Darwin: %s", archive_url)
    elif system == "Linux":
        archive_url = f"https://github.com/protocolbuffers/protobuf/releases/download/v3.11.4/protoc-3.11.4-linux-{machine}.zip"
        logging.info("Downloading protoc for Linux: %s", archive_url)
    else:
        archive_url = None
    return archive_url


def get_protoc():
    """make sure protoc is available, otherwise download it and return a tuple with the protoc
    binary and a temporary dir if it needed to be downloaded"""
    if not shutil.which("protoc"):
        archive_url = _protoc_bundle()
        if not archive_url:
            raise RuntimeError(
                "protoc not installed. Please install it manually by running sh protoc_downloader.sh"
            )
        logging.info("Downloading protoc")
        (fname, headers) = urllib.request.urlretrieve(archive_url)
        tmpdir = tempfile.mkdtemp(prefix="protoc_smdebug")
        with ZipFile(fname, "r") as zipf:
            zipf.extractall(tmpdir)
        protoc_bin = os.path.join(tmpdir, "bin", "protoc")
        os.chmod(protoc_bin, 0o755)
        return (protoc_bin, tmpdir)
    return (shutil.which("protoc"), None)


def compile_protobuf():
    """
    Compile protobuf files for smdebug
    """
    logging.info("Compile protobuf")
    logging.info("================")
    (protoc_bin, tmpdir) = get_protoc()
    cmd = [protoc_bin]
    proto_files = glob.glob("smdebug/core/tfevent/proto/*.proto")
    cmd.extend(proto_files)
    cmd.append("--python_out=.")
    logging.info("Call to protoc: %s", " ".join(cmd))
    check_call(cmd)
    if tmpdir:
        shutil.rmtree(tmpdir, ignore_errors=True)


def main():
    config_logging()
    compile_protobuf()
    return 0


if __name__ == "__main__":
    sys.exit(main())
