#!/usr/bin/env bash
if [ $1 == 'all' ]; then
    mypy @mypy_include.txt | awk -v pwd=$(pwd) '{print pwd "/" $0;}'
else
    mypy $1 | awk -v pwd=$(pwd) '{print pwd "/" $0;}'
fi
