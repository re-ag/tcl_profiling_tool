
import os
import sys
import math

from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message
from common import get_rosbag_options
from tcl_std_msgs.msg import ProfileData

import rosbag2_py

import numpy as np
import argparse

def get_shortcut_from_task_name(task_name):
    name = ''
    if task_name == 'virtual_front_lidar_driver':
        name = 'FRONT_LID'
    elif task_name == 'virtual_rear_lidar_driver':
        name = 'REAR_LID'
    elif task_name == 'filter_transform_vlp16_front':
        name = 'FRONT_PFT'
    elif task_name == 'filter_transform_vlp16_rear':
        name = 'REAR_PFT'
    elif task_name == 'point_cloud_fusion':
        name = 'PCF'
    elif task_name == 'scan_ground_filter':
        name = 'RGF'
    elif task_name =='euclidean_clustering':
        name = 'EUC'
    elif task_name == 'voxel_grid_downsample_filter':
        name = 'VGF'
    elif task_name == 'ndt_scan_matcher':
        name = 'NDT'
    elif task_name == 'virtual_camera_driver':
        name = 'CAM'
    elif task_name == 'tensorrt_yolo':
        name = 'TRT'
    elif task_name == 'vision_detections':
        name = 'VID'
    elif task_name == 'multi_object_tracker':
        name = 'MOT'    
    elif task_name == 'virtual_driver_vehicle_kinematic_state':
        name = 'ODM'
    elif task_name == 'ekf_localizer':
        name = 'EKF'
    elif task_name == 'behavior_planner':
        name = 'BHP'
    elif task_name == 'pure_pursuit':
        name = 'PPS'
    elif task_name == 'simulation/dummy_perception_publisher':
        name = 'DPP'
    elif task_name == 'simulation/detected_object_feature_remover':
        name = 'OFR'
    elif task_name == 'perception/object_recognition/tracking/multi_object_tracker':
        name = 'MOT'
    elif task_name == 'perception/object_recognition/prediction/map_based_prediciton':
        name = 'MBP'
    elif task_name == 'planning/scenario_planning/lane_driving/behavior_planning/behavior_path_planner':
        name = 'BPP'
    elif task_name == 'planning/scenario_planning/lane_driving/behavior_planning/behavior_velocity_planner':
        name = 'BVP'
    elif task_name == 'planning/scenario_planning/lane_driving/motion_planning/obstacle_avoidance_planner':
        name = 'OAP'
    elif task_name == 'planning/scenario_planning/lane_driving/motion_planning/obstacle_cruise_planner':
        name = 'OCP'
    elif task_name == 'planning/scenario_planning/motion_velocity_smoother':
        name = 'MVS'
    elif task_name == 'control/trajectory_follower/mpc_follower':
        name = 'MPC'
    else:
        name = ''
    return name

def top_of_cum_dist(cum_list, prob_th):
    """
    return index of 'cumulative_prob_list' that satisfies prob_list[index] >= '1-prob_th'
    """
    if prob_th < .0 or prob_th > 1.0:
        return None
    
    max_prob = np.max(cum_list)
    prob_th = min(1 - prob_th, max_prob)

    for idx in range(0, len(cum_list)):
        if cum_list[idx] >= prob_th:
            return idx
    return None

def make_hist(result_dict, hist_type):
    hist_dict = dict()

    for result in result_dict:
        name = result
        data = [int(round(x)) for x in result_dict[name]]
        hist = np.zeros(1000, dtype=int)
        for i in range(0, len(data)):
            hist[data[i]] = hist[data[i]]+1
        
        if hist_type == 'prob':
            denom = float(sum(hist))
            hist = [x / denom for x in hist]    
        hist = list(hist)
        hist_dict[name] = hist
    
    return hist_dict

def make_txt_result(result_dict, result_dir, result_type):

    f = None

    if result_type == 'et_prob':
        f = open(result_dir+'et_prob.txt','w')

    elif result_type == 'et_hist':
        f = open(result_dir+'et_hist.txt','w')

    elif result_type == 'e2e_prob':
        f = open(result_dir+'e2e_prob.txt','w')

    elif result_type == 'e2e_hist':
        f = open(result_dir+'e2e_hist.txt', 'w')

    elif result_type == 'rt_hist':
        f = open(result_dir+'rt_hist.txt','w')

    elif result_type == 'rt_prob':
        f = open(result_dir+'rt_prob.txt','w')

    elif result_type == 'et_orig':
        f = open(result_dir+'et_orig_list.txt','w')

    for result in result_dict:
        name = result
        data = result_dict[result]
        f.write(name+'='+str(data))
        f.write('\n')

    for result in result_dict:
        name = result
        sum = 0.0
        f1 = open(result_dir+name+'.csv', 'w')

        for item in result_dict[result]:
            f1.write(str(sum))
            f1.write('\n')
            sum += item
        

def raw_profile(dir_name, cpu_infos):

    task_result = dict()
    cpu_result = dict()

    # bag_name= os.listdir(dir_name)[0]

    # storage_options, converter_options = get_rosbag_options(dir_name + bag_name)

    storage_options, converter_options = get_rosbag_options('/root/shared_dir/rosbag/profile/' + dir_name)
    reader = rosbag2_py.SequentialReader()
    reader.open(storage_options, converter_options)

    topic_types = reader.get_all_topics_and_types()
    type_map = {topic_types[i].name: topic_types[i].type for i in range(len(topic_types))}    

    while reader.has_next():
        (topic, data, t) = reader.read_next()
        msg_type = get_message(type_map[topic])
        msg = deserialize_message(data, msg_type)
        if task_result.get(topic) == None:
            task_result[topic] = list()
        task_result[topic].append(msg) #태스크 단위로 profile_data 저장
                                       #################################################
                                       # task_result dictionary 구조
                                       # {
                                       #   task1/tcl_profile_data : [msg1, msg2 ...]
                                       #   task2/tcl_profile_data : [msg1, msg2 ...]
                                       #  ...
                                       # }
                                       ##################################################
    
    for cpu, tasks in enumerate(cpu_infos):
        for task in tasks:
            timing_topic =  '/'+ task + '/tcl_profile_data'
            # print(timing_topic)
            if task_result.get(timing_topic) != None:
                if cpu_result.get(cpu) == None:
                    cpu_result[cpu] = list()
                for msg in task_result[timing_topic]:
                    cpu_result[cpu].append([msg.timing_header, msg.execution_time.start, msg.execution_time.end, msg.release_time.start, msg.release_time.end])
        cpu_result[cpu].sort(key=lambda x:x[1])
                                                #task 가 구동되는 cpu 단위로 실행 시작 순서에 따라 profile_data 저장
                                                ###########################################################################
                                                # cpu_result 구조
                                                # {
                                                #   cpu1 : [cpu1_task1, cpu1_task2, cpu1_task1, cpu1_task2 ....]
                                                #   cpu2 : [cpu2_task1, cpu2_task2, cpu2_task1, cpu2_task2 ....]
                                                #   ...
                                                # }
                                                # 실행시간 분포, 지연시간 분포 생성 시 태스크들의 실행 시작 순서가 필요하기 때문에 정렬 필요
                                                ###########################################################################

    return cpu_result

def raw_profile_1(dir_name, cpu_infos):

    task_result = dict()
    cpu_result = dict()

    # bag_name= os.listdir(dir_name)[0]

    # storage_options, converter_options = get_rosbag_options(dir_name + bag_name)

    storage_options, converter_options = get_rosbag_options('/root/shared_dir/rosbag/profile/' + dir_name)
    reader = rosbag2_py.SequentialReader()
    reader.open(storage_options, converter_options)

    topic_types = reader.get_all_topics_and_types()
    type_map = {topic_types[i].name: topic_types[i].type for i in range(len(topic_types))}    

    while reader.has_next():
        (topic, data, t) = reader.read_next()
        msg_type = get_message(type_map[topic])
        msg = deserialize_message(data, msg_type)
        task_nm = msg.timing_header.task_name
        if task_result.get(task_nm) == None:
            task_result[task_nm] = list()
        task_result[task_nm].append(msg) #태스크 단위로 profile_data 저장
                                       #################################################
                                       # task_result dictionary 구조
                                       # {
                                       #   task1 : [msg1, msg2 ...]
                                       #   task2 : [msg1, msg2 ...]
                                       #  ...
                                       # }
                                       ##################################################

    for cpu, tasks in enumerate(cpu_infos):
        for task in tasks:
            # timing_topic =  '/'+ task + '/tcl_profile_data'
            # print(timing_topic)
            if task_result.get(task) != None:
                if cpu_result.get(cpu) == None:
                    cpu_result[cpu] = list()
                for msg in task_result[task]:
                    cpu_result[cpu].append([msg.timing_header, msg.execution_time.start, msg.execution_time.end, msg.release_time.start, msg.release_time.end])
        cpu_result[cpu].sort(key=lambda x:x[1])
                                                #task 가 구동되는 cpu 단위로 실행 시작 순서에 따라 profile_data 저장
                                                ###########################################################################
                                                # cpu_result 구조
                                                # {
                                                #   cpu1 : [cpu1_task1, cpu1_task2, cpu1_task1, cpu1_task2 ....]
                                                #   cpu2 : [cpu2_task1, cpu2_task2, cpu2_task1, cpu2_task2 ....]
                                                #   ...
                                                # }
                                                # 실행시간 분포, 지연시간 분포 생성 시 태스크들의 실행 시작 순서가 필요하기 때문에 정렬 필요
                                                ###########################################################################

    return cpu_result




def process_execution_time_orig(profile_data): #실행시간을 한 태스크의 시작 - 종료로 정의한 오리지널 버전
    
    execution_time_result = dict()

    for cpu, profile in profile_data.items():
        for data in profile:
            task_name = data[0].task_name
            elapse = (data[2] - data[1]) * 1.0e-6 # msg.execution_time.end - msg.execution_time.start
            elapse_ceil = math.ceil(elapse)

            if execution_time_result.get(task_name) == None:
                execution_time_result[task_name] = list()
            
            # print(task_name, elapse_ceil)
            execution_time_result[task_name].append(elapse_ceil) #task 단위로 execution time 의 history 저장
                                                                 ########################################
                                                                 # execution_time_result 구조
                                                                 # {
                                                                 #  task1 : [1, 5, 3, 4, 3, 2 ...]
                                                                 #  task2 : [12, 15, 13, 12, 14, 11 ...]
                                                                 #  ...
                                                                 # }
    
    for key, value in execution_time_result.items():
        print(key + ' execution time')
        print('max' , round(max(value)))
        print('avg' , round(sum(value)/len(value)))
        print('min' , round(min(value)))
        print('------')
    
    return execution_time_result

def process_execution_time_task_to_task(task_cpu_infos, profile_data): #실행시간을 선행 태스크 시작 - 후행 태스크 시작으로 정의한 버전
    
    execution_time_result = dict()
    print('----------------------------------------------------------------------------------------------------------------------------')
    for cpu, profile in profile_data.items():
        sink_task_in_cpu = task_cpu_infos[int(cpu)][-1]
        # print("sink_task: ", sink_task_in_cpu)
        drop_cnt = 0
        no_drop_cnt = 0
        a = 0
        # print(sink_task_in_cpu, len(profile))
        for i in range(0, len(profile)):
            drop = False
            timing_info = profile[i][0]
            task_name = timing_info.task_name
            
            elapse = (profile[i][2] - profile[i][1]) * 1.0e-6
            elapse_ceil = math.ceil(elapse)
            # if sink_task_in_cpu.find(task_name) == -1 and i+1 < len(profile):
            if sink_task_in_cpu != task_name and i+1 < len(profile):
                # drop = False
                # Make current task msg id list
                a += 1
                curr_mid_list = list()
                for minfo in profile[i][0].msg_infos:
                    curr_mid_list.append(minfo.msg_id)
                # Make next task msg id list
                next_mid_list = list()
                for minfo in profile[i+1][0].msg_infos:
                    next_mid_list.append(minfo.msg_id)
                # return
                
                
                curr_nm = profile[i][0].task_name
                next_nm = profile[i+1][0].task_name
                idx = task_cpu_infos[int(cpu)].index(curr_nm)
                if idx != (len(task_cpu_infos[int(cpu)]) -1):
                    orig_next_nm = task_cpu_infos[int(cpu)][idx+1]
                    if orig_next_nm != next_nm:
                        # print(curr_nm, orig_next_nm, next_nm)
                        drop = True
                
                for mid in curr_mid_list:
                    if mid not in next_mid_list: 
                        # print(curr_nm, curr_mid_list)
                        # print(next_nm, next_mid_list)
                        drop = True
                
                if drop == False: # drop 발생하지 않은 경우는 후행 태스크 - 선행 태스크
                    elapse = (profile[i+1][1] - profile[i][1]) * 1.0e-6
                    elapse_ceil = math.ceil(elapse)  
                    no_drop_cnt += 1
                    
                else: # drop 발생한 경우는 현재 태스크의 종료 - 현재 태스크의 시작 으로 계산함
                    drop_cnt += 1
                
            if execution_time_result.get(task_name) == None:
                execution_time_result[task_name] = list()
        
            execution_time_result[task_name].append(elapse_ceil) #task 단위로 execution time 의 history 저장
                                                                 ########################################
                                                                 # execution_time_result 구조
                                                                 # {
                                                                 #  task1 : [1, 5, 3, 4, 3, 2 ...]
                                                                 #  task2 : [12, 15, 13, 12, 14, 11 ...]
                                                                 #  ...
                                                                 # }
        # if drop_cnt > 0 :
        #     print(task_cpu_infos[int(cpu)][0], " to ", task_cpu_infos[int(cpu)][-1], "drop count is ", drop_cnt)
        print("%30s to %30s : drop (%6d) / total (%6d) " %(task_cpu_infos[int(cpu)][0], task_cpu_infos[int(cpu)][-1], drop_cnt, a))
        # if no_drop_cnt > 0:
        # print(task_cpu_infos[int(cpu)][0], " to ", task_cpu_infos[int(cpu)][-1], "  drop count : ", drop_cnt, " / no drop count : ", no_drop_cnt)

    print('----------------------------------------------------------------------------------------------------------------------------\n\n')
    for key, value in execution_time_result.items():
        print(key + ' execution time')
        print('max' , round(max(value)))
        print('avg' , round(sum(value)/len(value)))
        print('min' , round(min(value)))
        print('------')
    
    return execution_time_result

def process_response_time(profile_data):

    response_time_result = dict()

    for cpu, profile in profile_data.items():
        for data in profile:
            task_name = data[0].task_name
            elapse = (data[4] - data[3]) * 1.0e-6 # msg.response_time.end - msg.response_time.start

            if response_time_result.get(task_name) == None:
                response_time_result[task_name] = list()

            response_time_result[task_name].append(elapse) #task 단위로 response time 의 history 저장
                                                           ########################################
                                                           # response_time_result 구조
                                                           # {
                                                           #  task1 : [1, 5, 3, 4, 3, 2 ...]
                                                           #  task2 : [12, 15, 13, 12, 14, 11 ...]
                                                           #  ...
                                                           # }
    
    for key, value in response_time_result.items():
        print(key + ' response time')
        print('max' , round(max(value)))
        print('avg' , round(sum(value)/len(value)))
        print('min' , round(min(value)))
        print('------')

    return response_time_result

def process_e2e_latency_using_source_sink(profile_data, paths):

    e2e_latency_result = dict()

    for path in paths:
        source_task_name = path[0]
        sink_task_name = path[-1]
        source_task_cpu = None
        sink_task_cpu = None

        objective_path_str = str(path) #관심 경로

        sink_result = dict()

        for cpu, profile in profile_data.items():
            for data in profile:
                if data[0].task_name == source_task_name:
                    source_task_cpu = cpu
                if data[0].task_name == sink_task_name:
                    sink_task_cpu = cpu
                if source_task_cpu is not None and sink_task_cpu is not None:
                    break 
        #profile data 가 cpu 단위로 저장되기 때문에 source, sink task 의 cpu 번호 저장        

        for data in profile_data[sink_task_cpu]:
            if data[0].task_name == sink_task_name:
                for msg_info in data[0].msg_infos:
                    task_history_set = set(msg_info.task_history)
                    path_set = set(path)
                    if path_set.intersection(task_history_set) == path_set: #timing info 에 objective_path 가 존재하는지
                        sink_result[msg_info.msg_id] = data[4]   
                        break                                   #msg_id 단위로 sink task 의 response_time.end 저장
                                                                ########################################
                                                                # sink_result 구조
                                                                # {
                                                                #  1 : [165423434.12351231]
                                                                #  2 : [165423434.15431230]
                                                                #  ...
                                                                # }   

        for data in profile_data[source_task_cpu]:
            if data[0].task_name == source_task_name:
                for msg_info in data[0].msg_infos:
                    if sink_result.get(msg_info.msg_id) is not None: #sink 와 source 의 msg_id 가 같으면
                        elapse = (sink_result[msg_info.msg_id] - data[0].msg_infos.creation_time) * 1.0e-6 #sink_task response_time.end - source task response_time.start (or msg create time)
                        if e2e_latency_result.get(objective_path_str) == None:
                            e2e_latency_result[objective_path_str] = list()
                        e2e_latency_result[objective_path_str].append(elapse)
                        break                                                #object_path 단위로 e2e latency history 저장
                                                                             ########################################
                                                                             # e2e_latency_result 구조
                                                                             # {
                                                                             #  obj_path1 : [32.1, 35.23, 30.11 ....]
                                                                             #  obj_path2 : [78.21, 77.09, 80.43 ....]
                                                                             #  ...
                                                                             # }  
                            
    for key, value in e2e_latency_result.items():
        print(key + ' e2e latency')
        print('max' , round(max(value)))
        print('min' , round(min(value)))
        print('avg' , round(sum(value)/len(value)))
        print('------')     

    return e2e_latency_result

def process_e2e_latency_using_sink(profile_data, paths):

    e2e_latency_result = dict()

    for key, path in paths.items():
        sink_task_name = path[-1]
        # print(sink_task_name)
        sink_task_cpu = None

        objective_path_str = str(path) #관심 경로

        sink_result = dict()

        msg_ids = []
        for cpu, profile in profile_data.items():
            for data in profile:
                if data[0].task_name == sink_task_name:
                    sink_task_cpu = cpu
                if sink_task_cpu is not None:
                    break 
        # print(sink_task_cpu)
        #profile data 가 cpu 단위로 저장되기 때문에 source, sink task 의 cpu 번호 저장        
        cnt = 0
        for data in profile_data[sink_task_cpu]:
            if data[0].task_name == sink_task_name:
                for msg_info in data[0].msg_infos:
                    task_history_set = set(msg_info.task_history)
                    path_set = set(path)
                    if path_set.intersection(task_history_set) == path_set: #timing info 에 objective_path 가 존재하는지
                        cnt += 1
                        elapse = (data[2] - msg_info.creation_time) * 1.0e-6
                        # elapse = (data[2] - msg_info.creation_time)
                        if e2e_latency_result.get(key) == None:
                            e2e_latency_result[key] = list()
                        if msg_info.msg_id not in msg_ids:
                            e2e_latency_result[key].append(elapse)   
                            msg_ids.append(msg_info.msg_id)
                        break                                   #msg_id 단위로 sink task 의 response_time.end 저장
                                                                ########################################
                                                                # sink_result 구조
                                                                # {
                                                                #  1 : [165423434.12351231]
                                                                #  2 : [165423434.15431230]
                                                                #  ...
                                                                # }  
        # print("path exists... ", cnt)                 
    for key, value in e2e_latency_result.items():
        print(key + ' e2e latency')
        print('max' , round(max(value)))
        print('min' , round(min(value)))
        print('avg' , round(sum(value)/len(value)))
        print('------')     

    return e2e_latency_result

def cpu_utilization(task_cpu_infos, execution_time_data):
    period = [100, 50, 50, 50, 50, 100, 10, 100, 100, 100, 100, 100]
    for cpu,tasks in enumerate(task_cpu_infos):
        avg_ = 0
        max_ = 0
        for task in tasks:
            index = task.rfind('/')
            if index == -1:
                task_nm = task
            else:
                task_nm = task[index+1:]
            print(task_nm, "avg: ", round((sum(execution_time_data[task_nm])/len(execution_time_data[task_nm]))), "ms max: ", max(execution_time_data[task_nm]), "ms")
            avg_ += round((sum(execution_time_data[task_nm])/len(execution_time_data[task_nm])))
            max_ += max(execution_time_data[task_nm])
        print('#### CPU', (cpu+1),  'avg: ', avg_/ period[cpu] * 100.0, '% max: ', max_ / period[cpu] * 100.0,'% \n')





def main(args):
    input_dir = args.input
    output_dir = args.output
    orig = args.orig
    # 태스크 - 코어 할당 정보
    # cpu0 = ['virtual_rear_lidar_driver', 'lidar/perception/concatenate_filter', 'lidar/perception/crop_box_filter', 'lidar/perception/lidar_centerpoint']
    # cpu1 = ['virtual_camera_driver', 'cam/perception/image_transport_decompressor']
    # cpu2 = ['cam/perception/tensorrt_yolo', 'cam/perception/roi_detected_object_fusion']
    # cpu3 = ['cam/perception/multi_object_tracker']
    # cpu4 = ['cam/perception/map_based_prediction']
    # cpu5 = ['virtual_front_lidar_driver', 'lidar/localization/voxel_grid_downsample_filter', 'lidar/localization/ndt_scan_matcher']
    # # cpu5 = ['virtual_front_lidar_driver', 'lidar/perception/concatenate_filter', 'lidar/localization/voxel_grid_downsample_filter', 'lidar/localization/ndt_scan_matcher']
    # cpu6 = ['virtual_can_driver', 'odom/localization/vehicle_velocity_converter', 'odom/localization/ekf_localizer_ndt', 'odom/localization/stop_filter']
    # cpu7 = ['planning/behavior_path_planner']
    # cpu8 = ['planning/behavior_velocity_planner']
    # cpu9 = ['planning/obstacle_avoidance_planner']
    # cpu10 = ['planning/obstacle_cruise_planner']
    # cpu11 = ['planning/motion_velocity_smoother']


    cpu0 = ['virtual_rear_lidar_driver', 'concatenate_filter', 'crop_box_filter', 'lidar_centerpoint']
    cpu1 = ['virtual_camera_driver', 'image_transport_decompressor']
    # cpu2 = ['tensorrt_yolo']
    cpu2 = ['tensorrt_yolo', 'roi_detected_object_fusion']
    cpu3 = ['multi_object_tracker']
    cpu4 = ['map_based_prediction']
    cpu5 = ['virtual_front_lidar_driver', 'voxel_grid_downsample_filter', 'ndt_scan_matcher']
    cpu6 = ['virtual_can_driver', 'vehicle_velocity_converter', 'ekf_localizer_ndt', 'stop_filter']
    cpu7 = ['behavior_path_planner']
    cpu8 = ['behavior_velocity_planner']
    cpu9 = ['obstacle_avoidance_planner']
    cpu10 = ['obstacle_cruise_planner']
    cpu11 = ['motion_velocity_smoother']
    # task_cpu_infos = [cpu0, cpu1, cpu2, cpu3, cpu4, cpu5, cpu6]
    # task_cpu_infos = [cpu0, cpu1, cpu2, cpu3, cpu4, cpu5, cpu6]
    task_cpu_infos = [cpu0, cpu1, cpu2, cpu3, cpu4, cpu5, cpu6, cpu7, cpu8, cpu9, cpu10, cpu11]
    # task_cpu_infos = [cpu0, cpu1, cpu2, cpu3, cpu4, cpu5, cpu6]
    task_cpu_infos = [cpu6]


    # e2e latency 를 얻고자하는 관심 경로 (경로 상의 모든 태스크 입력)
    e2e_paths = {
        # 'CAM_DET' : ['virtual_camera_driver', 'image_transport_decompressor', 'tensorrt_yolo', 'roi_detected_object_fusion', 'multi_object_tracker', 'map_based_prediction'],
        # 'FRONT_LIDAR_DET': ['virtual_front_lidar_driver', 'concatenate_filter', 'crop_box_filter', 'lidar_centerpoint', 'roi_detected_object_fusion', 'multi_object_tracker', 'map_based_prediction'],
        # 'REAR_LIDAR_DET': ['virtual_rear_lidar_driver', 'concatenate_filter', 'crop_box_filter', 'lidar_centerpoint', 'roi_detected_object_fusion', 'multi_object_tracker', 'map_based_prediction'],
        # 'FRONT_LIDAR_LOC' :['virtual_front_lidar_driver', 'voxel_grid_downsample_filter', 'ndt_scan_matcher', 'ekf_localizer_ndt', 'stop_filter'],
        # 'ODOM' : ['virtual_can_driver', 'vehicle_velocity_converter', 'ekf_localizer_ndt', 'stop_filter'],
        # 'CAM_DET' : ['virtual_camera_driver', 'image_transport_decompressor', 'tensorrt_yolo', 'roi_detected_object_fusion', 'multi_object_tracker', 'map_based_prediction', 'behavior_path_planner', 'behavior_velocity_planner', 'obstacle_avoidance_planner', 'obstacle_cruise_planner', 'motion_velocity_smoother'],
        # 'FRONT_LIDAR_DET': ['virtual_front_lidar_driver', 'concatenate_filter', 'crop_box_filter', 'lidar_centerpoint', 'roi_detected_object_fusion', 'multi_object_tracker', 'map_based_prediction', 'behavior_path_planner', 'behavior_velocity_planner', 'obstacle_avoidance_planner', 'obstacle_cruise_planner', 'motion_velocity_smoother'],
        # 'REAR_LIDAR_DET': ['virtual_rear_lidar_driver', 'concatenate_filter', 'crop_box_filter', 'lidar_centerpoint', 'roi_detected_object_fusion', 'multi_object_tracker', 'map_based_prediction', 'behavior_path_planner', 'behavior_velocity_planner', 'obstacle_avoidance_planner', 'obstacle_cruise_planner', 'motion_velocity_smoother'],
        # 'FRONT_LIDAR_LOC' :['virtual_front_lidar_driver', 'voxel_grid_downsample_filter', 'ndt_scan_matcher', 'ekf_localizer_ndt', 'stop_filter', 'behavior_path_planner', 'behavior_velocity_planner', 'obstacle_avoidance_planner', 'obstacle_cruise_planner', 'motion_velocity_smoother'],
        # 'ODOM' : ['virtual_can_driver', 'vehicle_velocity_converter', 'ekf_localizer_ndt', 'stop_filter', 'behavior_path_planner', 'behavior_velocity_planner', 'obstacle_avoidance_planner', 'obstacle_cruise_planner', 'motion_velocity_smoother'],
        # 'SUB_CAM_DET' : ['virtual_camera_driver', 'image_transport_decompressor', 'tensorrt_yolo', 'roi_detected_object_fusion', 'multi_object_tracker', 'map_based_prediction'],
        # 'SUB_FRONT_LIDAR_DET': ['virtual_front_lidar_driver', 'concatenate_filter', 'crop_box_filter', 'lidar_centerpoint'],
        # 'SUB_REAR_LIDAR_DET': ['virtual_rear_lidar_driver', 'concatenate_filter', 'crop_box_filter', 'lidar_centerpoint'],
        # 'SUB_FRONT_LIDAR_LOC' :['virtual_front_lidar_driver', 'voxel_grid_downsample_filter', 'ndt_scan_matcher'],
        'SUB_ODOM' : ['virtual_can_driver', 'vehicle_velocity_converter', 'ekf_localizer_ndt', 'stop_filter'],
        # 'SUB_PLANNING': ['behavior_path_planner', 'behavior_velocity_planner', 'obstacle_avoidance_planner', 'obstacle_cruise_planner', 'motion_velocity_smoother']
        # 'SUB_PLANNING': ['behavior_path_planner']
    }

    # profile_data = raw_profile(input_dir, task_cpu_infos) #rosbag 을 읽고, cpu 단위로 task 들의 profile data 저장
    profile_data = raw_profile_1(input_dir, task_cpu_infos) #rosbag 을 읽고, cpu 단위로 task 들의 profile data 저장
    if(orig):
        execution_time_data = process_execution_time_orig(profile_data) #profile_data 를 사용하여 task 들의 execution_time 분포 획득
                                                                    #실행시간을 한 태스크의 시작 - 종료로 정의한 오리지널 버전
    else:
        execution_time_data = process_execution_time_task_to_task(task_cpu_infos, profile_data) #profile_data 를 사용하여 task 들의 execution_time 분포 획득
                                                                            # 실행시간을 선행 태스크의 시작 - 후행 태스크의 시작으로 정의한 버전
    
    # return

    e2e_latency_data    = process_e2e_latency_using_sink(profile_data, e2e_paths) #profile_data 를 사용하여 e2e_path 의 e2e_latency 분포 획득
    # return
    execution_time_prob = make_hist(execution_time_data, 'prob') #각 task 의 execution time 확률 분포 획득
    e2e_latency_prob = make_hist(e2e_latency_data, 'prob') #각 관심 경로의 e2e latency 확률 분포 획득
    
    make_txt_result(execution_time_prob, output_dir, 'et_prob') #execution time 확률 분포를 txt 파일로 저장    
    make_txt_result(e2e_latency_prob, output_dir, 'e2e_prob') #e2e latency 확률 분포를 txt 파일로 저장
    # cpu_utilization(task_cpu_infos, execution_time_data)

    ###response time###
    # response_time_data  = process_response_time(profile_data) #profile_data 를 사용하여 task 들의 response_time 분포 획득
    # response_time_prob = make_hist(response_time_data, 'prob') #각 task 의 response time 확률 분포 획득
    # make_txt_result(response_time_prob, output_dir, 'rt_prob') #response time 확률 분포를 txt 파일로 저장

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', help='input profiled directory name', required=True)
    parser.add_argument('-o', '--output', help='output result directory name', required=True)
    parser.add_argument('-orig', '--orig', action='store_true')

    args = parser.parse_args()

    main(args)