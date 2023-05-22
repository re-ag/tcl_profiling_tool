#!/bin/bash

last_file=$(ls -al | tail -1 | awk -F' ' '{print $9}')

ros2 bag info $last_file

echo $last_file
# ros2 bag info $(ls -al | tail -1 | awk -F' ' '{print $9}')

