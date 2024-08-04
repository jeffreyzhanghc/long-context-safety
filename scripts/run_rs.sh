#!/bin/bash -i
time python -u exps/rs.py --shots $2 
#> ./logs/rs_$2/$1.txt 2>&1