import os
import centernet
import argparse
import numpy as np
import cv2
import shutil
from collections import defaultdict
import json


from sort import Sort
from lib import VisTrack, Camera, CameraFront


def main (camerasDic):
    # config variables
    max_age, min_hits, iou_threshold = 2, 3, 0.3
    duration = 5 # time in seconds
    skip_detect = 5 # doing object detection every n frames, to not skip on any frame: skip_detect = 1
    desired_interval = 2 # taking every n frames, to not skip on any frame: desired_interval = 1
    sort = Sort(max_age, min_hits, iou_threshold)
    obj = centernet.ObjectDetection(num_classes=80)
    obj.load_weights(weights_path=None) # type: ignore
    mapping_ids = {}
    stay_durations_dic = defaultdict(set)
    result = {}
    video_path = []
    time_counter = 0
    seconds_counter = 0
    ids_per_second = defaultdict(set)
    people_per_second = {}

    

    # create the res for each camera
    for camera in camerasDic.values():
        camera.create_res(duration=duration, desired_interval=desired_interval, skip_detect=skip_detect,sort=sort,obj=obj)

    max_resDic_length = max(len(camera.resDic) for camera in camerasDic.values())

    # update the cameras detections
    for frame in range(max_resDic_length): # run on range (0,number of frames)
        for camera in camerasDic.values():
            if frame in camera.resDic:
                if not camera.updated[frame]:
                    mapping_ids = camera.update_frame(frame= frame, camerasDic= camerasDic, mapping_ids= mapping_ids)
                for index, res in enumerate(camera.resDic[frame]):
                    id = int(res[-1])
                    stay_durations_dic[id].add(frame)
                    if frame % (camera.fps//desired_interval) == 0:
                        ids_per_second[seconds_counter].add(id)
        if frame % (camera.fps//desired_interval) == 0:
            people_per_second[seconds_counter] = len(ids_per_second[seconds_counter])
            seconds_counter += 1

    # draw the res on frames
    for folder in [name for name in camerasDic]:
        if os.path.exists(f'Track/{folder}'):
            shutil.rmtree(f'Track/{folder}')
        os.makedirs(f'Track/{folder}')
    vt = VisTrack()
    for camera in camerasDic.values():
        camera.draw_bounding_boxes(duration,desired_interval,vt,f'Track/{camera.name}')
    
    #create a videos
    for camera in camerasDic.values():
        camera.create_video(frames_dir=f'Track/{camera.name}',output_file= f'dataset/marked/{camera.name}.mp4',desired_interval=desired_interval)
        video_path.append(f'dataset/marked/{camera.name}.mp4')
        

    #calcualate iou avg
    average = np.mean(sort.list_of_iou)
    print("The avg of iou is:", average)

    # print the duration time for each id
    print(f'The number of people in the store was {len(stay_durations_dic)}')
    first_camera = next(iter(camerasDic.values()))
    fps = first_camera.fps
    for id, frames in stay_durations_dic.items():
        duration_in_sec = (len(frames) * desired_interval) / fps
        time_counter += duration_in_sec
        print (f'ID number {id} stayed {duration_in_sec:.2f} seconds')
        result[id] = f'{duration_in_sec:.2f}'

    total_customers = len(stay_durations_dic)

    avg_time = time_counter/len(stay_durations_dic)
   
    return video_path, result, total_customers, avg_time, people_per_second

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate ids for object tracking.')
    parser.add_argument('-s', '--source', type=str, help='Path to the json source file')

    args = parser.parse_args()
    source = args.source

    with open(source, 'r') as f:
        data = json.load(f)

    cameras = data['cameras']
    front = cameras['front']
    store = cameras['store']
    camerasDic = {}

    camerasDic[front["name"]] = CameraFront(name=front["name"], vidoePath=front["path"], overlappingDic=front["overlapping"])

    for camera in store:
        camerasDic[camera["name"]] = Camera(name=camera["name"], vidoePath=camera["path"], overlappingDic=camera["overlapping"])

    result = main(camerasDic) 
    print(result)
   