import os
import centernet
import time
import argparse
import numpy as np
import PIL.Image
import cv2
from tqdm.auto import tqdm
import shutil
from collections import defaultdict


from sort import Sort
from lib import VisTrack, Camera, CameraFront

def split_paths(path_string):
    names = path_string.split(',')
    front = names[0].strip()
    store = names[1].strip()

    return front, store


if __name__ == "__main__":

    # Create the argument
    parser = argparse.ArgumentParser(description='Generate ids for object tracking.')
    parser.add_argument('-g', '--generate-frames', action='store_true', default=False, help='Generate frames.')
    parser.add_argument('-c', '--create-videos', action='store_true', default=False, help='Create video.')
    parser.add_argument('-m', '--calculate-metrics', action='store_true', default=False, help='calculate the avg of ious.')
    parser.add_argument('-s', '--source', type=str, help='Path to the video source file')
    parser.add_argument('-d', '--destination', type=str, help='Path to the video destination file')


    args = parser.parse_args()
    generate_frames = args.generate_frames
    create_videos = args.create_videos
    calculate_avg_iou = args.calculate_metrics
    source = args.source
    destination = args.destination
    
    print("generate_frames:", generate_frames)
    print("create_video:", create_videos)
    print("calcualte_avg_iou:", calculate_avg_iou)
    print("source:", source)
    print("destination:", destination)

    # config variables
    video_file_f, video_file_s = split_paths(source) if source is not None else ("./dataset/videos/front_2.mp4", "./dataset/videos/store_2.mp4")
    output_file_f, output_file_s = split_paths(destination) if destination is not None else ("dataset/Track-front.mp4", "dataset/Track-store.mp4")
    folder_out_front = "Track/Track-front"
    folder_out_store = "Track/Track-store"
    folder_outs = [folder_out_front,folder_out_store]
    max_age, min_hits, iou_threshold = 2, 3, 0.3
    duration = 10 # time in seconds
    skip_detect = 5 # doing object detection every n frames, to not skip on any frame: skip_detect = 1
    desired_interval = 2 # taking every n frames, to not skip on any frame: desired_interval = 1
    sort = Sort(max_age, min_hits, iou_threshold)
    obj = centernet.ObjectDetection(num_classes=80)
    obj.load_weights(weights_path=None) # type: ignore

    cameraF = CameraFront(name='front', vidoePath=video_file_f, overlappingDic={'store': np.array([230,640,450,1080]),
                                                                           'door': np.array([440,630,510,990])})
    
    cameraS = Camera(name='store', vidoePath=video_file_s, overlappingDic={'front': np.array([440,630,570,1080])})
    
    if generate_frames:
        cameraF.create_res(duration=duration, desired_interval=desired_interval, skip_detect=skip_detect,sort=sort,obj=obj)
        cameraS.create_res(duration=duration, desired_interval=desired_interval, skip_detect=skip_detect,sort=sort,obj=obj)

        mapping_ids = {}
        camerasDic = {
            cameraF.name: cameraF,
            cameraS.name: cameraS
        }

        stay_durations_dic = defaultdict(set)

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

        # draw the res on frames
        for folder in folder_outs:
            if os.path.exists(folder):
                shutil.rmtree(folder)
            os.makedirs(folder)
        vt = VisTrack()
        cameraF.draw_bounding_boxes(duration,desired_interval,vt,folder_out_front)
        cameraS.draw_bounding_boxes(duration,desired_interval,vt,folder_out_store)
    
    #calcualate iou avg
    if calculate_avg_iou:
        average = np.mean(sort.list_of_iou)
        print("The avg of iou is:", average)

    #create a videos
    if create_videos:
        cameraF.create_vidoe(frames_dir=folder_out_front,output_file= "dataset/Track-front.mp4",frame_size= (608,1080),desired_interval=desired_interval)
        cameraF.create_vidoe(frames_dir=folder_out_store,output_file= "dataset/Track-store.mp4",frame_size= (608,1080),desired_interval=desired_interval)

    # print the duration time for each id
    print(f'The number of pepole in the store was {len(stay_durations_dic)}')
    vidcap = cv2.VideoCapture(cameraF.vidoePath) # need to figure how to save the fps to vidoe
    fps = vidcap.get(cv2.CAP_PROP_FPS)
    for id, frames in stay_durations_dic.items():
        duration_in_sec = (len(frames) * desired_interval) / fps
        print (f'ID number {id} stay {duration_in_sec:.2f} seconds')
