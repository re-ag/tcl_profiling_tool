
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
os.sched_setaffinity( 0, [14])

task_to_cpu_= {
    'virtual_camera_driver' : 50,
    'image_transport_decompressor' : 50,
    'tensorrt_yolo' : 50,
    'roi_detected_object_fusion' : 50,
    'multi_object_tracker' : 50,
    'map_based_prediction' : 50,
    'virtual_rear_lidar_driver': 100,
    'concatenate_filter': 100,
    'crop_box_filter': 100,
    'lidar_centerpoint': 100,
    'virtual_front_lidar_driver': 100,
    'voxel_grid_downsample_filter': 100,
    'ndt_scan_matcher': 100,
    'virtual_can_driver' : 10,
    'vehicle_velocity_converter' : 10,
    'ekf_localizer_ndt' : 10,
    'stop_filter' : 10,
    'behavior_path_planner' : 100,
    'behavior_velocity_planner' : 100,
    'obstacle_avoidance_planner' : 100,
    'obstacle_cruise_planner' : 100,
    'motion_velocity_smoother' : 100,
}


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
        # f = open(result_dir+'et_prob.txt','w')
        f = open(result_dir+'etd_list.py','w')

    elif result_type == 'et_hist':
        f = open(result_dir+'et_hist.txt','w')

    elif result_type == 'e2e_prob':
        f = open(result_dir+'ld_list.py','w')

    elif result_type == 'e2e_hist':
        f = open(result_dir+'e2e_hist.txt', 'w')

    elif result_type == 'rt_hist':
        f = open(result_dir+'rt_hist.txt','w')

    elif result_type == 'rt_prob':
        f = open(result_dir+'rtd_list.py','w')

    elif result_type == 'et_orig':
        f = open(result_dir+'et_orig_list.txt','w')

    for result in result_dict:
        name = result
        data = result_dict[result]
        f.write(name+'='+str(data))
        f.write('\n')

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
        if len(msg.timing_header) != 0:
            task_nm = msg.timing_header[0].task_name
        else:
            continue
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
                    cpu_result[cpu].append([list(msg.timing_header), msg.execution_time.start, msg.execution_time.end, msg.release_time.start, msg.release_time.end])
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
    print("End read rosbag")
    return task_result, cpu_result




def process_execution_time_orig(profile_data): #실행시간을 한 태스크의 시작 - 종료로 정의한 오리지널 버전
    
    execution_time_result = dict()

    for cpu, profile in profile_data.items():
        for data in profile:
            task_name = data[0][0].task_name
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

def find_index(lists, value):
    for idx, lst in enumerate(lists):
        # if val == value:
        #     return idx
        for val in lst:
            if val == value:
                return idx
    return -1

def process_execution_time_task_to_task(task_cpu_infos, profile_data, sub_paths, task_results): #실행시간을 선행 태스크 시작 - 후행 태스크 시작으로 정의한 버전
    margin = 10
    execution_time_result = dict()
    no_dro_cnt = dict()
    print('----------------------------------------------------------------------------------------------------------------------------')
    first = True
    for cpu, profile in profile_data.items():
        sink_task_in_cpu = task_cpu_infos[int(cpu)][-1]
        # print("current_cpu: " ,cpu)
        
        for i in range(0, len(profile)):
            timing_headers = profile[i][0]
            curr_name = timing_headers[0].task_name

            if no_dro_cnt.get(curr_name) == None:
                no_dro_cnt[curr_name] = 0
            sub_idx = -1
            sub_idx = find_index(sub_paths, curr_name)
            
            
            curr_mid_list = list()
            curr_pub_list = list()
            for timing_info in timing_headers:
                for minfo in timing_info.msg_infos:
                    task_history = list(minfo.task_history)
                    sub = list(sub_paths[sub_idx])
                    if sub[0] == task_history[0] and timing_info.published:
                    # if sub[0] == task_history[0]:
                        curr_mid_list.append(minfo.msg_id)
                        curr_pub_list.append(timing_info.published)
            # print(curr_mid_list)
            elapse = (profile[i][2] - profile[i][1]) * 1.0e-6
            elapse_ceil = math.ceil(elapse)
            yeah = False
        
            pidx = 0
            if sink_task_in_cpu != curr_name:
                idx = task_cpu_infos[int(cpu)].index(curr_name)
                next_name = task_cpu_infos[int(cpu)][idx+1]
                # print("orig next " , next_name)
                # for pi, p in enumerate(profile):
                sidx = 0
                if i - margin < 0:
                    sidx = 0
                else:
                    sidx = i - margin
                for pi in range(sidx, len(profile)):
                    if profile[pi][0][0].task_name == next_name:
                        # print("p[0].task_name", p[0].task_name)
                        next_sub_idx = find_index(sub_paths, next_name)
                        next_mid_list = list()
                        for header in profile[pi][0]:
                            for minfo in header.msg_infos:
                                task_history = list(minfo.task_history)
                                nex_sub = list(sub_paths[next_sub_idx])
                                if task_history[0] == nex_sub[0]:
                                    next_mid_list.append(minfo.msg_id)
                        # print(curr_mid_list, next_mid_list)
                        for m_idx, mid in enumerate(curr_mid_list):
                            if mid in next_mid_list and curr_pub_list[m_idx] == True:
                                ## execution time 계산하기
                                    
                                yeah = True
                                elapse = (profile[pi][1] - profile[i][1] ) * 1.0e-6
                                elapse_ceil = math.ceil(elapse)
                                # if(curr_name == 'behavior_path_planner' and elapse_ceil == 37):
                                #     print("behavior_path_planner", mid, profile[i][1], profile[i][2])
                                #     print("\n\nnext_header")
                                #     print(profile[pi][0])
                                #     print(profile[pi][1])
                                # pidx = pi
                                no_dro_cnt[curr_name] += 1
                                break
                    if yeah:
                        break
                if elapse_ceil < 0:
                    print('--------------------------------------------------------------------------------------')
                    print("ELAPSE IS NEGATIVE!!!!!!!!!!  " << elapse)
                    print(profile[i][0],profile[i][1])
                    print(profile[pi][0], profile[pi][1])
            # if pidx != 0 :
                # print(pidx, len(profile))
            if execution_time_result.get(curr_name) == None:
                execution_time_result[curr_name] = list()
            if elapse_ceil > 0:
                execution_time_result[curr_name].append(elapse_ceil)
            else:
                print(curr_name, elapse_ceil)
    
    for cpu in task_cpu_infos:
        sink = cpu[-1]
        for task in cpu:
            if task != sink:
                print("%30s drop count : %6d / %6d"%(task, len(task_results[task]) - no_dro_cnt.get(task), len(task_results[task])))        

    # for name, cnt in no_dro_cnt.items():
    #     print("%30s : %6d / %6d"%(name, cnt, len(task_results[name])))



    execution_time_result = dict(sorted(execution_time_result.items()))
    # print('----------------------------------------------------------------------------------------------------------------------------\n\n')
    # for key, value in execution_time_result.items():
    #     print('%s execution time: (%3d / %3d / %3d)'%(key.center(30), min(value), sum(value)/len(value), max(value)))
        # print(key + ' execution time')
        # print('min' , round(min(value)))
        # print('avg' , round(sum(value)/len(value)))
        # print('max' , round(max(value)))
        # print('------')
        # print('----------------------------------------------------------------------------------------------------------------------------')
        
    
    return execution_time_result

def process_response_time(profile_data):

    response_time_result = dict()
    # print("Response time -----------------------------------------------------------------------------------------------------------")
    for cpu, profile in profile_data.items():
        for data in profile:
            task_name = data[0][0].task_name
            elapse = (data[4] - data[3]) * 1.0e-6 # msg.response_time.end - msg.response_time.start
            elapse_ceil = math.ceil(elapse)
            # if(elapse_ceil >= 290):
            #     print(data[0])
            if response_time_result.get(task_name) == None:
                response_time_result[task_name] = list()
            if(elapse_ceil >= 500 and (task_name.find("planner") != -1 or task_name.find("smoother")) ):
                print(data[0])
                for th in data[0]:
                    for minfo in th.msg_infos:
                        print("msg_id = " , minfo.msg_id)
                print(data[1], data[2], data[3], data[4])
                # break
            response_time_result[task_name].append(elapse_ceil) #task 단위로 response time 의 history 저장
                                                           ########################################
                                                           # response_time_result 구조
                                                           # {
                                                           #  task1 : [1, 5, 3, 4, 3, 2 ...]
                                                           #  task2 : [12, 15, 13, 12, 14, 11 ...]
                                                           #  ...
                                                           # }
    response_time_result = dict(sorted(response_time_result.items()))
    # for key, value in response_time_result.items():
    #     print('%s response time: (%3d / %3d / %3d)'%(key.center(30), min(value), sum(value)/len(value), max(value)))
        # print(key + ' response time')
        # print('min' , round(min(value)))
        # print('avg' , round(sum(value)/len(value)))
        # print('max' , round(max(value)))
        # print('------')
        # print('----------------------------------------------------------------------------------------------------------------------------')


    return response_time_result

def process_e2e_latency_using_source_sink(profile_data, paths):

    e2e_latency_result = dict()

    for key, path in paths.items():
        source_task_name = path[0]
        sink_task_name = path[-1]
        source_task_cpu = None
        sink_task_cpu = None

        objective_path_str = str(path) #관심 경로

        sink_result = dict()

        for cpu, profile in profile_data.items():
            for data in profile:
                if data[0][0].task_name == source_task_name:
                    source_task_cpu = cpu
                if data[0][0].task_name == sink_task_name:
                    sink_task_cpu = cpu
                if source_task_cpu is not None and sink_task_cpu is not None:
                    break 
        #profile data 가 cpu 단위로 저장되기 때문에 source, sink task 의 cpu 번호 저장        

        for data in profile_data[sink_task_cpu]:
            if data[0][0].task_name == sink_task_name:
                for header in data[0]:
                    for msg_info in header.msg_infos:
                        task_history_set = set(msg_info.task_history)
                        path_set = set(path)
                        if path_set.intersection(task_history_set) == path_set : #timing info 에 objective_path 가 존재하는지
                            # print(task_history_set)
                            sink_result[msg_info.msg_id] = data[2]   
                            break                                   #msg_id 단위로 sink task 의 execution_time.end 저장
                                                                    ########################################
                                                                    # sink_result 구조
                                                                    # {
                                                                    #  1 : [165423434.12351231]
                                                                    #  2 : [165423434.15431230]
                                                                    #  ...
                                                                    # }   

        for data in profile_data[source_task_cpu]:
            if data[0][0].task_name == source_task_name:
                for header in data[0]:
                    if header.published:
                        for msg_info in header.msg_infos:
                            if sink_result.get(msg_info.msg_id) is not None: #sink 와 source 의 msg_id 가 같으면
                                # elapse = (sink_result[msg_info.msg_id] - data[0].msg_infos.creation_time) * 1.0e-6 #sink_task response_time.end - source task response_time.start (or msg create time)
                                elapse = (sink_result[msg_info.msg_id] - data[1]) * 1.0e-6
                                if e2e_latency_result.get(key) == None:
                                    e2e_latency_result[key] = list()
                                e2e_latency_result[key].append(elapse)
                                if elapse < 0:
                                    print(msg_info.msg_id, sink_result[msg_info.msg_id], data[1])
                                break                                                #object_path 단위로 e2e latency history 저장
                                                                                 ########################################
                                                                                 # e2e_latency_result 구조
                                                                                 # {
                                                                                 #  obj_path1 : [32.1, 35.23, 30.11 ....]
                                                                                 #  obj_path2 : [78.21, 77.09, 80.43 ....]
                                                                                 #  ...
                                                                                 # }  
                            
    # for key, value in e2e_latency_result.items():
    #     print('%s : (%3d / %3d / %3d)' %((key+' e2e latency').center(40), round(min(value)), round(sum(value)/len(value)), round(max(value))))
    #     print('---------------------------------------------------------------------')
        # print(key + ' e2e latency')
        # print('min' , round(min(value)))
        # print('avg' , round(sum(value)/len(value)))
        # print('max' , round(max(value)))
        # print('------')     

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
                if data[0][0].task_name == sink_task_name:
                    sink_task_cpu = cpu
                if sink_task_cpu is not None:
                    break 
        # print(sink_task_cpu)
        #profile data 가 cpu 단위로 저장되기 때문에 source, sink task 의 cpu 번호 저장        
        cnt = 0
        for data in profile_data[sink_task_cpu]:
            if data[0][0].task_name == sink_task_name:
                for header in data[0]:
                    for msg_info in header.msg_infos:
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
    # for key, value in e2e_latency_result.items():
    #     print('%s : (%3d / %3d / %3d)' %((key+' e2e latency').center(40), round(min(value)), round(sum(value)/len(value)), round(max(value))))
    #     print('---------------------------------------------------------------------')

    return e2e_latency_result

def print_time_result(execution_time_data, response_time_data, task_cpu_infos):
    for cpu in task_cpu_infos:
        for task in cpu:
            print(task)
            print('execution  time: (%3d / %3d / %3d)'%(min(execution_time_data[task]), round(sum(execution_time_data[task])/len(execution_time_data[task])), max(execution_time_data[task])))
            print('response   time: (%3d / %3d / %3d)'%(min(response_time_data[task]), round(sum(response_time_data[task])/len(response_time_data[task])), max(response_time_data[task])))
            print('---------------------------------------------------------------------')
            
def print_latency(e2e_latency):
    for key, value in e2e_latency.items():
        print('%s : (%3d / %3d / %3d)' %((key+' e2e latency').center(40), round(min(value)), round(sum(value)/len(value)), round(max(value))))
        print('---------------------------------------------------------------------')

def cpu_utilization(task_cpu_infos, execution_time_data):
    # period = [100, 100, 50, 50, 50, 10, 10, 100]
    print("CPU UTILIZATION")
    for cpu,tasks in enumerate(task_cpu_infos):
        min_ = 0
        avg_ = 0
        max_ = 0
        period_ = task_to_cpu_[tasks[0]]
        for task in tasks:
            print('%s ( %.0lf, %.0lf, %.0lf )'%(task.center(30), min(execution_time_data[task]), sum(execution_time_data[task])/len(execution_time_data[task]), max(execution_time_data[task])))
            # print(task, "avg: ", round((sum(execution_time_data[task])/len(execution_time_data[task]))), "ms max: ", max(execution_time_data[task]), "ms")
            min_ += min(execution_time_data[task])
            avg_ += (sum(execution_time_data[task])/len(execution_time_data[task]))
            max_ += max(execution_time_data[task])
        # print('#### CPU', (cpu+1),  'avg: ', avg/ period[cpu] * 100.0, '% max: ', max / period[cpu] * 100.0,'% \n')
        print('## CPU %d utilization (min / avg / max) : (%0.1f%%, %0.1f%%, %0.1f%% )\n'%(cpu+1, min_/period_ *100.0, avg_/period_ * 100.0, max_/period_ * 100.0))





def main(args):
    input_dir = args.input
    output_dir = args.output
    orig = args.orig
    # 태스크 - 코어 할당 정보


    # cpu0 = ['virtual_rear_lidar_driver', 'concatenate_filter', 'crop_box_filter', 'lidar_centerpoint']
    # # cpu1 = ['lidar_centerpoint']
    # cpu1 = ['virtual_front_lidar_driver', 'voxel_grid_downsample_filter', 'ndt_scan_matcher']
    # cpu2 = ['virtual_camera_driver', 'image_transport_decompressor'] 
    # cpu3 = ['tensorrt_yolo', 'roi_detected_object_fusion','multi_object_tracker', 'map_based_prediction']
    # cpu4 = ['virtual_can_driver','vehicle_velocity_converter']
    # cpu5 = ['ekf_localizer_ndt', 'stop_filter']
    # cpu6= ['behavior_path_planner','behavior_velocity_planner']
    # cpu7 = ['obstacle_avoidance_planner', 'obstacle_cruise_planner', 'motion_velocity_smoother']

    # task_cpu_infos = [cpu0, cpu1, cpu2, cpu3, cpu4, cpu5, cpu6, cpu7]
    cpu0 = ['virtual_rear_lidar_driver', 'concatenate_filter', 'crop_box_filter']
    cpu1 = ['lidar_centerpoint']
    cpu2 = ['virtual_front_lidar_driver', 'voxel_grid_downsample_filter', 'ndt_scan_matcher']
    cpu3 = ['virtual_camera_driver', 'image_transport_decompressor'] 
    cpu4 = ['tensorrt_yolo']
    cpu5 = ['roi_detected_object_fusion','multi_object_tracker', 'map_based_prediction']
    cpu6 = ['virtual_can_driver','vehicle_velocity_converter']
    cpu7 = ['ekf_localizer_ndt', 'stop_filter']
    cpu8 = ['behavior_path_planner']
    cpu9 = ['behavior_velocity_planner']
    cpu10 = ['obstacle_avoidance_planner', 'obstacle_cruise_planner', 'motion_velocity_smoother']

    task_cpu_infos = [cpu0, cpu1, cpu2, cpu3, cpu4, cpu5, cpu6, cpu7, cpu8, cpu9, cpu10]

    execution_time_data = None
    response_time_data = None
    e2e_latency_data = None

    sub_paths = [
        ['virtual_camera_driver', 'image_transport_decompressor', 'tensorrt_yolo', 'roi_detected_object_fusion', 'multi_object_tracker', 'map_based_prediction'],
        ['virtual_rear_lidar_driver', 'concatenate_filter', 'crop_box_filter', 'lidar_centerpoint'],
        ['virtual_front_lidar_driver', 'voxel_grid_downsample_filter', 'ndt_scan_matcher'],
        ['virtual_can_driver', 'vehicle_velocity_converter', 'ekf_localizer_ndt', 'stop_filter'],
        ['behavior_path_planner', 'behavior_velocity_planner', 'obstacle_avoidance_planner', 'obstacle_cruise_planner', 'motion_velocity_smoother']
        
    ]
    # e2e latency 를 얻고자하는 관심 경로 (경로 상의 모든 태스크 입력)
    e2e_paths = {
        'CAM_DET' : ['virtual_camera_driver', 'image_transport_decompressor', 'tensorrt_yolo', 'roi_detected_object_fusion', 'multi_object_tracker', 'map_based_prediction', 'behavior_path_planner', 'behavior_velocity_planner', 'obstacle_avoidance_planner', 'obstacle_cruise_planner', 'motion_velocity_smoother'],
        'FLD_DET': ['virtual_front_lidar_driver', 'concatenate_filter', 'crop_box_filter', 'lidar_centerpoint', 'roi_detected_object_fusion', 'multi_object_tracker', 'map_based_prediction', 'behavior_path_planner', 'behavior_velocity_planner', 'obstacle_avoidance_planner', 'obstacle_cruise_planner', 'motion_velocity_smoother'],
        'RLD_DET': ['virtual_rear_lidar_driver', 'concatenate_filter', 'crop_box_filter', 'lidar_centerpoint', 'roi_detected_object_fusion', 'multi_object_tracker', 'map_based_prediction', 'behavior_path_planner', 'behavior_velocity_planner', 'obstacle_avoidance_planner', 'obstacle_cruise_planner', 'motion_velocity_smoother'],
        'FLD_LOC' :['virtual_front_lidar_driver', 'voxel_grid_downsample_filter', 'ndt_scan_matcher', 'ekf_localizer_ndt', 'stop_filter', 'behavior_path_planner', 'behavior_velocity_planner', 'obstacle_avoidance_planner', 'obstacle_cruise_planner', 'motion_velocity_smoother'],
        'ODM_LOC' : ['virtual_can_driver', 'vehicle_velocity_converter', 'ekf_localizer_ndt', 'stop_filter', 'behavior_path_planner', 'behavior_velocity_planner', 'obstacle_avoidance_planner', 'obstacle_cruise_planner', 'motion_velocity_smoother'],
        'SUB_CAM_DET' : ['virtual_camera_driver', 'image_transport_decompressor', 'tensorrt_yolo', 'roi_detected_object_fusion', 'multi_object_tracker', 'map_based_prediction'],
        'SUB_FLD_DET': ['virtual_front_lidar_driver', 'concatenate_filter', 'crop_box_filter', 'lidar_centerpoint'],
        'SUB_RLD_DET': ['virtual_rear_lidar_driver', 'concatenate_filter', 'crop_box_filter', 'lidar_centerpoint'],
        'SUB_FLD_LOC' :['virtual_front_lidar_driver', 'voxel_grid_downsample_filter', 'ndt_scan_matcher'],
        'SUB_ODM_LOC' : ['virtual_can_driver', 'vehicle_velocity_converter', 'ekf_localizer_ndt', 'stop_filter'],
        'SUB_PLANNING': ['behavior_path_planner', 'behavior_velocity_planner', 'obstacle_avoidance_planner', 'obstacle_cruise_planner', 'motion_velocity_smoother']
    }
    planning_paths = {   
        'SUB_PLANNING': ['behavior_path_planner', 'behavior_velocity_planner', 'obstacle_avoidance_planner', 'obstacle_cruise_planner', 'motion_velocity_smoother']
    }
    task_results, profile_data = raw_profile(input_dir, task_cpu_infos) #rosbag 을 읽고, cpu 단위로 task 들의 profile data 저장
    # return
    if(orig):
        execution_time_data = process_execution_time_orig(profile_data) #profile_data 를 사용하여 task 들의 execution_time 분포 획득
                                                                    #실행시간을 한 태스크의 시작 - 종료로 정의한 오리지널 버전
    else:
        execution_time_data = process_execution_time_task_to_task(task_cpu_infos, profile_data, sub_paths, task_results) #profile_data 를 사용하여 task 들의 execution_time 분포 획득
                                                                            # 실행시간을 선행 태스크의 시작 - 후행 태스크의 시작으로 정의한 버전
    cpu_utilization(task_cpu_infos, execution_time_data)
    # return
    response_time_data  = process_response_time(profile_data) #profile_data 를 사용하여 task 들의 response_time 분포 획득
    print_time_result(execution_time_data, response_time_data, task_cpu_infos)
    # return

    e2e_latency_data    = process_e2e_latency_using_sink(profile_data, e2e_paths) #profile_data 를 사용하여 e2e_path 의 e2e_latency 분포 획득
    # sub_planning_latency_data = process_e2e_latency_using_source_sink(profile_data, planning_paths)
    # e2e_latency_data.update(sub_planning_latency_data)
    print_latency(e2e_latency_data)


    execution_time_prob = make_hist(execution_time_data, 'prob') #각 task 의 execution time 확률 분포 획득
    e2e_latency_prob = make_hist(e2e_latency_data, 'prob') #각 관심 경로의 e2e latency 확률 분포 획득
    response_time_prob = make_hist(response_time_data, 'prob') #각 task 의 response time 확률 분포 획득

    make_txt_result(response_time_prob, output_dir, 'rt_prob') #response time 확률 분포를 txt 파일로 저장
    make_txt_result(execution_time_prob, output_dir, 'et_prob') #execution time 확률 분포를 txt 파일로 저장    
    make_txt_result(e2e_latency_prob, output_dir, 'e2e_prob') #e2e latency 확률 분포를 txt 파일로 저장

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', help='input profiled directory name', required=True)
    parser.add_argument('-o', '--output', help='output result directory name', required=True)
    parser.add_argument('-orig', '--orig', action='store_true')

    args = parser.parse_args()

    main(args)