#!/bin/bash

check_env CASHMERE_PORT

# The configuration arguments needs to be a string without any whitespace,
# separated by commas.  This will be split in a later stage.

if echo "$@" | grep -q -- "-cpu"
then
    config_args="-Dcashmere.nLocalExecutors=16,-Xmx50G,-Dibis.constellation.stealing=mw"
elif echo "$@" | grep -q -- "-mc"
then
    config_args="-Dcashmere.nLocalExecutors=4,-Xmx5G,-Dibis.constellation.stealing=mw"
elif echo "$@" | grep -q -- "-mainMemCache"
then
    config_args="-Dcashmere.nLocalExecutors=4,-Xmx5G,-Dibis.constellation.stealing=mw"
elif echo "$@" | grep -q -- "-deviceMemCache"
then
    config_args="-Dcashmere.nLocalExecutors=4,-Xmx5G"
elif echo "$@" | grep -q -- "-remote-activities""
then
    config_args="-Dcashmere.nLocalExecutors=4,-Xmx5G"
elif echo "$@" | grep -q -- "-dedicated-activities""
then
    config_args="-Dcashmere.nLocalExecutors=4,-Xmx5G"
else
    echo "Need parameter -cpu, -mc, -mainMemCache, -deviceMemCache, -remote-activities or -dedicated-activities"
    exit 1
fi
