#!/usr/bin/env bash

readarray -t files_to_lint < mypy_include.txt
pylint -j 3 --rcfile /home/stephen/projects/mqc/pylintrc "${files_to_lint[@]}"
