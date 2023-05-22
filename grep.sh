#!/bin/bash
latest_file=$(ls -al *.txt --sort=time | head -1 | awk -F' ' '{print $9}')

echo $latest_file
cat $latest_file | grep -e "camera_driver" -e "image_transport" -e "tensorrt_yolo" -e "roi_detect" -e "multi_object" -e "map_based" > /autoware/src/cam_pipe$latest_file
cat $latest_file | grep -e "front_lidar_driver" -e "voxel_grid" -e "ndt_scan" > /autoware/src/fld_pipe$latest_file
cat $latest_file | grep -e "rear_lidar_driver" -e "concatenate_filter_node" -e "crop_box_filter_node" -e "lidar_centerpoint" > /autoware/src/rld_pipe$latest_file
cat $latest_file | grep -e "can_driver" -e "vehicle_velocity_converter" -e "ekf_localizer" -e "stop_filter" > /autoware/src/odm_pipe$latest_file
cat $latest_file | grep -e "behavior_path" -e "behavior_velocity_planner" -e "obstacle_avoidance" -e "obstacle_cruise" -e "motion_velocity" > /autoware/src/pln_pipe$latest_file

