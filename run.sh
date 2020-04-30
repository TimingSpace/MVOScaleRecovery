#!/bin/bash
for d in $(cat $1) ; do
    echo $d
    python src/main.py $d
done
