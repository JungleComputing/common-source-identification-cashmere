#!/bin/bash

check_env CASHMERE_PORT

extra_args=( )
if echo "$@" | grep -q -- "-cpu"
then
    extra_args+=("-Dcashmere.nLocalExecutors=16")
    extra_args+=("-Xmx50G")
elif echo "$@" | grep -q -- "-mc"
then
    extra_args+=("-Dcashmere.nLocalExecutors=4")
    extra_args+=("-Xmx5G")
elif echo "$a" | grep -q -- "-use_cache"
then
    extra_args+=("-Dcashmere.nLocalExecutors=4")
    extra_args+=("-Xmx5G")
else
    echo "Need parameter -cpu, -mc, or -use_cache"
    exit 1
fi
