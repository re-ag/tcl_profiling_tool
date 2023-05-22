#!/bin/bash

date

ros2 bag record $(ros2 topic list | grep tcl_profile_data)

