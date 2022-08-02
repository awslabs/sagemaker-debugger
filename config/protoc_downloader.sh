#!/bin/bash
unameOut="$(uname -s)"
case "${unameOut}" in
    Linux*)     machine=Linux;;
    Darwin*)    machine=Mac;;
esac
if [ "$machine" = "Mac" ] ; then
    PROTOC_ZIP=protoc-3.7.1-osx-x86_64.zip
    brew install unzip
else
    PROTOC_ZIP=protoc-3.7.1-linux-x86_64.zip
    apt-get install sudo
    sudo apt-get install unzip
fi
curl -OL https://github.com/google/protobuf/releases/download/v3.7.1/$PROTOC_ZIP
unzip -o $PROTOC_ZIP -d /usr/local bin/protoc
unzip -o $PROTOC_ZIP -d /usr/local include/*
rm -f $PROTOC_ZIP
