#!/usr/bin/env bash

# user_defined parameters  sequence - 0: file location, 1: is_html[True, False], 2:text_stemming[True, False], 3:ft_dim, 4: N-cluster
#
python3 ./app/main.py ./input/Papers.csv False True 300 10