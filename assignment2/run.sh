#!/bin/bash
question=$1

if [[ ${question} == "1" ]]; then
    path_of_train_data=$2
    path_of_test_data=$3
    part_num=$4
    python3 Q1/q1.py --train_path $path_of_train_data --test_path $path_of_test_data --part $part_num
fi

if [[ ${question} == "2" ]]; then
    path_of_train_data=$2
    path_of_test_data=$3
    binary_or_multi_class=$4
    part_num=$5
    if [[ ${binary_or_multi_class} == "0" ]]; then
        python3 Q2/q2_a.py --train_path $path_of_train_data --test_path $path_of_test_data --part $part_num
    fi
    if [[ ${binary_or_multi_class} == "1" ]]; then
        python3 Q2/q2_b.py --train_path $path_of_train_data --test_path $path_of_test_data --part $part_num
    fi
fi
