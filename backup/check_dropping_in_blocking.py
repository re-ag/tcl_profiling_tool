import os
import argparse

import rosbag2_py
from common import get_rosbag_options
from rosidl_runtime_py.utilities import get_message
from rclpy.serialization import deserialize_message

def read_bag(dir):

    task_result = dict()
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
        # print(topic_nm[1:-17])
        msg_type = get_message(type_map[topic_nm])
        msg = deserialize_message(data, msg_type)
        task_nm = msg.timing_header.task_name
        if task_result.get(task_nm) == None:
            task_result[task_nm] = list() 
        for minfo in msg.timing_header.msg_infos:
            task_result[task_nm].append(minfo.msg_id)
    return task_result

def read_bag_per_msg(dir):

    task_result = dict()
    msg_result = dict()
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
        # print(topic_nm[1:-17])
        msg_type = get_message(type_map[topic_nm])
        msg = deserialize_message(data, msg_type)
        task_nm = msg.timing_header.task_name
        if task_result.get(task_nm) == None:
            task_result[task_nm] = list() 
        if msg_result.get(task_nm) == None:
            msg_result[task_nm] = list()
        task_result[task_nm].append(msg.timing_header.msg_infos[0].msg_id)
        msg_result[task_nm].append(msg)
    return task_result, msg_result

def check_drop(result, paths):
    # lst = result['virtual_camera_driver']
    # for i in range(len(lst)):
    #     if lst[i].timing_header.msg_infos[0].msg_id == 167842891926678:
    #         print(lst[i])
    #         print(lst[i+1])
    # return
    
    # for idx, mid in enumerate(result['virtual_can_driver']):
    #     if mid not in result['vehicle_velocity_converter']:
    #         print(idx, mid)
    # return

    print("============================= Check message duplication in source tasks =============================")
    curr = 0
    cnt = 0
    for path in paths:
        src = path[0]
        if(src.find('driver') == -1):
            continue
        curr = 0
        cnt = 0
        for mid in result[src]:
            if curr == mid:
                cnt += 1
            curr = mid
        if (cnt > 0):
            print(src + ": " + str(cnt))
    
    print("=====================================================================================================")

    for path in paths:
        curr = 0
        print("------------------------------------------------------------------------------------------------------------------------")
        print("PATH: " + path[0] + " to " + path[-1])
        
        for task in path:
            print(task +"(" + str(len(result[task])) + ") - ", end='')
        print('\n')

        for i in range(1, len(path)):
            # for j in range(3,len(result[path[i-1]])-3):
            for j in range(0,len(result[path[i-1]])):
                if result[path[i-1]][j] not in result[path[i]]:
                    print(str(j) + "th " + str(result[path[i-1]][j]) + " Message was dropped from " + path[i-1] + " to " + path[i])

                        
                    
    print("------------------------------------------------------------------------------------------------------------------------")    


def main(args):
    input_dir = args.input
    task_result = read_bag(input_dir)
    # task_result, msg_result = read_bag_per_msg(input_dir)
    # return
    # tasks = ['virtual_camera_driver', 'image_transport_decompressor', 'tensorrt_yolo', 'virtual_front_lidar_driver', 'virtual_rear_lidar_driver', 'concatenate_filter', 'crop_box_filter', 'lidar_centerpoint', 'roi_detected_object_fusion', 'multi_object_tracker', 'map_based_prediction', 'vehicle_velocity_converter', 'ekf_localizer_ndt', 'stop_filter']
    # return
    e2e_paths = [
        ['virtual_camera_driver', 'image_transport_decompressor', 'tensorrt_yolo', 'roi_detected_object_fusion', 'multi_object_tracker', 'map_based_prediction'],
        ['virtual_front_lidar_driver', 'concatenate_filter', 'crop_box_filter', 'lidar_centerpoint', 'roi_detected_object_fusion', 'multi_object_tracker', 'map_based_prediction'],
        ['virtual_rear_lidar_driver', 'concatenate_filter', 'crop_box_filter', 'lidar_centerpoint', 'roi_detected_object_fusion', 'multi_object_tracker', 'map_based_prediction'],
        ['virtual_front_lidar_driver', 'voxel_grid_downsample_filter', 'ndt_scan_matcher','ekf_localizer_ndt', 'stop_filter'],
        ['virtual_can_driver', 'vehicle_velocity_converter', 'ekf_localizer_ndt', 'stop_filter'],
        # ['behavior_path_planner','behavior_velocity_planner', 'obstacle_avoidance_planner', 'obstacle_cruise_planner', 'motion_velocity_smoother'],
        # ['behavior_path_planner'],
        
    ]
    check_drop(task_result, e2e_paths)

            



            

            

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', help='rosbag directory name', required=True)
    args = parser.parse_args()


    main(args)