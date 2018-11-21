#!/usr/bin/env bash

# user_defined parameters  sequence - 1: file location, 2:is_html[True, False], 3:text_stemming[True, False], 4:ft_dim, 5: N-cluster
#
# Parm 2, 3 - These are the Parameters for text pre-processing
# Parm 4 - This is the parameter for Fasttext dimension
# Parm 5 - This is parameter for the number of cluster needed in task 3
#
python3 ./app/main.py ./input/Papers.csv False True 300 10