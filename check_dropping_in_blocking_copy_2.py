import os
import argparse

import rosbag2_py
from common import get_rosbag_options
from rosidl_runtime_py.utilities import get_message
from rclpy.serialization import deserialize_message

def read_bag(dir, paths):

    task_result = dict()
    path_result = dict()
    published_result = dict()
    # bag_name = os.listdir(dir)[0]
    # storage_options, converter_options = get_rosbag_options(dir + bag_name)
    
    # bag_name = os.listdir(dir)
    storage_options, converter_options = get_rosbag_options('/root/shared_dir/rosbag/profile/' + dir)
    reader = rosbag2_py.SequentialReader()
    reader.open(storage_options, converter_options)

    topic_types = reader.get_all_topics_and_types()
    type_map = {topic_types[i].name: topic_types[i].type for i in range(len(topic_types))} 
    
    while reader.has_next():
        (topic_nm, data, t) = reader.read_next()
        msg_type = get_message(type_map[topic_nm])
        msg = deserialize_message(data, msg_type)
        task_nm = msg.timing_header[0].task_name
        # print(task_nm)
        if task_result.get(task_nm) == None:
            task_result[task_nm] = list() 
        task_result[task_nm].append(msg)

    # print(task_result['tensorrt_yolo'][0])
    for idx, path in enumerate(paths):
        for task in path:
            if task_result.get(task) != None:
                if published_result.get(idx) == None:
                    published_result[idx] = dict()
                if published_result[idx].get(task) == None:
                    published_result[idx][task] = list()

                if path_result.get(idx) == None:
                    path_result[idx] = dict()
                if path_result[idx].get(task) == None:
                    path_result[idx][task] = list()

                for msg in task_result[task]:
                    for header in msg.timing_header:
                        for minfo in header.msg_infos:
                            task_history = list(minfo.task_history)
                            if task_history[0] == path[0]:
                                mid = minfo.msg_id
                        path_result[idx][task].append(mid)
                        published_result[idx][task].append(header.published)
    
    # for idx, path in enumerate(paths):
    #     for task in path:
    #         if len(path_result[idx][task]) != len(list(set(path_result[idx].get(task)))):
    #             print(task, " original : ", len(path_result[idx][task]), " no duplicate : ", len(list(set(path_result[idx].get(task)))) )
    #     print("-------------------------------------------------------------------------------")

    return path_result, published_result

def check_drop__(path_results, published_results, paths):
    for p_idx, path in enumerate(paths):
        print("=======================================================================================================================================================================================")
        print(path, end='\n\n')
        for t_idx, curr_task in enumerate(path):
            cnt = 0
            for m_idx, mid in enumerate(path_results[p_idx][curr_task]):
                if t_idx < len(path) -1:
                    # if mid not in path_results[p_idx][path[t_idx+1]] and (m_idx > 2 and m_idx < len(path_results[p_idx][curr_task]) - 2 ):
                    if mid not in path_results[p_idx][path[t_idx+1]] and (m_idx > 2 and m_idx < len(path_results[p_idx][curr_task]) - 2 ) and published_results[p_idx][curr_task][m_idx]:
                        cnt += 1
                        # print(m_idx, " th / ", len(path_results[p_idx][curr_task])," ", mid, curr_task, " to " , path[t_idx+1])
            if t_idx < len(path) -1:
                # print(cnt, " / " , len(path_results[p_idx][curr_task]), " : ", curr_task, " to ", path[t_idx+1])
                print("%30s (%6d) to %30s (%6d) ---> drop count : %6d "%(curr_task.center(30), published_results[p_idx][curr_task].count(True), path[t_idx+1].center(30), len(path_results[p_idx][path[t_idx+1]]), cnt))
    print("=======================================================================================================================================================================================")
      

def main(args):
    input_dir = args.input
    sub_paths = [
        ['virtual_camera_driver', 'image_transport_decompressor', 'tensorrt_yolo', 'roi_detected_object_fusion', 'multi_object_tracker', 'map_based_prediction', 'behavior_path_planner','behavior_velocity_planner', 'obstacle_avoidance_planner', 'obstacle_cruise_planner', 'motion_velocity_smoother'],
        ['virtual_front_lidar_driver', 'concatenate_filter', 'crop_box_filter', 'lidar_centerpoint', 'roi_detected_object_fusion', 'multi_object_tracker', 'map_based_prediction', 'behavior_path_planner','behavior_velocity_planner', 'obstacle_avoidance_planner', 'obstacle_cruise_planner', 'motion_velocity_smoother'],
        ['virtual_rear_lidar_driver', 'concatenate_filter', 'crop_box_filter', 'lidar_centerpoint', 'roi_detected_object_fusion', 'multi_object_tracker', 'map_based_prediction', 'behavior_path_planner','behavior_velocity_planner', 'obstacle_avoidance_planner', 'obstacle_cruise_planner', 'motion_velocity_smoother'],
        ['virtual_front_lidar_driver', 'voxel_grid_downsample_filter', 'ndt_scan_matcher','ekf_localizer_ndt', 'stop_filter', 'behavior_path_planner','behavior_velocity_planner', 'obstacle_avoidance_planner', 'obstacle_cruise_planner', 'motion_velocity_smoother'],
        ['virtual_can_driver', 'vehicle_velocity_converter', 'ekf_localizer_ndt', 'stop_filter', 'behavior_path_planner','behavior_velocity_planner', 'obstacle_avoidance_planner', 'obstacle_cruise_planner', 'motion_velocity_smoother'],
        # ['virtual_camera_driver', 'image_transport_decompressor', 'tensorrt_yolo', 'roi_detected_object_fusion', 'multi_object_tracker', 'map_based_prediction', 'behavior_path_planner'],
        # ['virtual_front_lidar_driver', 'concatenate_filter', 'crop_box_filter', 'lidar_centerpoint', 'roi_detected_object_fusion', 'multi_object_tracker', 'map_based_prediction', 'behavior_path_planner'],
        # ['virtual_rear_lidar_driver', 'concatenate_filter', 'crop_box_filter', 'lidar_centerpoint', 'roi_detected_object_fusion', 'multi_object_tracker', 'map_based_prediction', 'behavior_path_planner'],
        # ['virtual_front_lidar_driver', 'voxel_grid_downsample_filter', 'ndt_scan_matcher','ekf_localizer_ndt', 'stop_filter', 'behavior_path_planner'],
        # ['virtual_can_driver', 'vehicle_velocity_converter', 'ekf_localizer_ndt', 'stop_filter', 'behavior_path_planner'],
    ]
    path_results, published_results = read_bag(input_dir, sub_paths)
    # return
    check_drop__(path_results, published_results, sub_paths)

            



            

            

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', help='rosbag directory name', required=True)
    args = parser.parse_args()


    main(args)