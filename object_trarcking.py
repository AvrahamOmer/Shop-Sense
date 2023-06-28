import os
import centernet
import time
import argparse
import numpy as np
import PIL.Image
import cv2
from tqdm.auto import tqdm
import shutil

from sort import Sort
from lib import VisTrack, Camera, CameraFront

if __name__ == "__main__":

    # Create the argument
    parser = argparse.ArgumentParser(description='Generate ids for object tracking.')
    parser.add_argument('-g', '--generate-frames', action='store_true', default=False, help='Generate frames.')
    parser.add_argument('-c', '--create-videos', action='store_true', default=False, help='Create video.')
    args = parser.parse_args()
    generate_frames = args.generate_frames
    create_videos = args.create_videos
    print("generate_frames:", generate_frames)
    print("create_video:", create_videos)

    # config variables
    video_file_f = "./dataset/videos/front_2.mp4"
    video_file_s = "./dataset/videos/store_2.mp4"
    folder_out_front = "Track/Track-front"
    folder_out_store = "Track/Track-store"
    folder_outs = [folder_out_front,folder_out_store]
    max_age, min_hits, iou_threshold = 2, 3, 0.3
    duration = 25 # time in seconds
    skip_detect = 5 # doing object detection every n frames, to not skip on any frame: skip_detect = 1
    desired_interval = 2 # taking every n frames, to not skip on any frame: desired_interval = 1
    sort = Sort(max_age, min_hits, iou_threshold)
    obj = centernet.ObjectDetection(num_classes=80)
    obj.load_weights(weights_path=None)

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

        # update the cameras detections
        for frame in range(len(cameraF.resDic)): # run on range (0,number of frames)
            for spot,camera in camerasDic.items():
                if frame in camera.resDic:
                    mapping_ids = camera.update_frame(frame= frame, camerasDic= camerasDic, mapping_ids= mapping_ids)

        # draw the res on frames
        for folder in folder_outs:
            if os.path.exists(folder):
                shutil.rmtree(folder)
            os.makedirs(folder)
        vt = VisTrack()
        cameraF.draw_bounding_boxes(duration,desired_interval,vt,folder_out_front)
        cameraS.draw_bounding_boxes(duration,desired_interval,vt,folder_out_store)

    #create a videos
    if create_videos:
        cameraF.create_vidoe(frames_dir=folder_out_front,output_file= "dataset/Track-front.mp4",frame_size= (608,1080),desired_interval=desired_interval)
        cameraF.create_vidoe(frames_dir=folder_out_store,output_file= "dataset/Track-store.mp4",frame_size= (608,1080),desired_interval=desired_interval)
