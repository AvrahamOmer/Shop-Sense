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
from lib import VisTrack, create_video, match_ID, add_fifth_axis


def generate_tracked_frames(video_files,folder_outs, duration, desired_interval, overlapping, skip_detect, max_age=1, min_hits=1, iou_threshold=0.3):

    dic_front = {}
    mapping_ids = {}
    sort = Sort(max_age, min_hits, iou_threshold)
    vt = VisTrack()
    id_object = 1 



    # Default: num_classes=80
    obj = centernet.ObjectDetection(num_classes=80)

    # num_classes=80 and weights_path=None: Pre-trained COCO model will be loaded.
    obj.load_weights(weights_path=None)

    for folder in folder_outs:
        if os.path.exists(folder):
            shutil.rmtree(folder)
        os.makedirs(folder)

    for video in video_files:
        vidcap = cv2.VideoCapture(video)
        fps = vidcap.get(cv2.CAP_PROP_FPS)
        est_tot_frames = int(duration * fps)  # Sets an upper bound # of frames in video clip
        frame_count = 0

        pbar = tqdm(total = int(est_tot_frames // desired_interval))
        out_put_file = folder_out1 if video == video_file1 else folder_out2
        start_time = time.time()  # Record the start time
        for i in range(0,est_tot_frames,desired_interval):
            vidcap.set(cv2.CAP_PROP_POS_FRAMES, i)
            sucsses, frame = vidcap.read()
            if not sucsses:
                break

            # do detection every 10 frames
            if (frame_count % skip_detect <= min_hits):
                boxes, classes, scores = obj.predict(frame)
                detections_in_frame = len(boxes)
                if detections_in_frame:
                    # centernet will do detection on all the COCO classes. "person" is class number 0 
                    idxs = np.where(classes == 0)[0]
                    boxes = boxes[idxs]
                    scores = scores[idxs]
                else:
                    boxes = np.empty((0, 4))
            else:
                if len(res):
                    boxes = res[:,:-1]
                else:
                    boxes = np.empty((0, 4))
                    
            dets = add_fifth_axis(boxes)
            res = sort.update(dets)

            print ("------------------")
            print(f'boxes in frame {i}: {boxes}')
            print(f'dets in frame {i}: {dets}')
            print (f'res in frame {i}: {res}')

            # mapping ids between front and store
            if video == video_file1:
                for k in range(len(res)):
                    id = int(res[k,-1])
                    if id not in mapping_ids:
                        mapping_ids[id] = id_object
                        id_object += 1
                    res[k,-1] = mapping_ids[id]
                dic_front[i] = res
            else:
                for j in range(len(res)):
                    id = int(res[j,-1])
                    if id  not in mapping_ids:
                        match = match_ID(overlapping,dic_front[i])
                        if match == 0: print (f'ERROR accur in frame {i}')
                        mapping_ids[id] = match
                    res[j,-1] = mapping_ids[id]

            boxes_track = res[:,:-1]
            boces_ids = res[:,-1].astype(int)

            p_frame = PIL.Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if detections_in_frame:
                p_frame = vt.draw_bounding_boxes(p_frame, boxes_track, boces_ids, scores)
            p_frame.save(os.path.join(out_put_file, f"{i:03d}.png"))

            pbar.update(1)
            frame_count += 1
        end_time = time.time()  # Record the end time
        execution_time = end_time - start_time  # Calculate the execution time
        print(f'Execution video {video} in {execution_time//60} mins')


if __name__ == "__main__":

    # Create the argument
    parser = argparse.ArgumentParser(description='Generate ids for object tracking.')
    parser.add_argument('-g', '--generate-frames', action='store_false', default=True, help='Generate frames.')
    parser.add_argument('-c', '--create-videos', action='store_false', default=True, help='Create video.')
    args = parser.parse_args()
    generate_frames = args.generate_frames
    create_videos = args.create_videos
    print("generate_frames:", generate_frames)
    print("create_video:", create_videos)

    # config variables
    video_file1 = "./dataset/videos/front_2.mp4"
    video_file2 = "./dataset/videos/store_2.mp4"
    folder_out1 = "Track/Track-front"
    folder_out2 = "Track/Track-store"
    folder_outs = [folder_out1, folder_out2]
    video_files = [video_file1, video_file2]
    overlapping = np.array([230,640,450,1080])
    max_age, min_hits, iou_threshold = 1, 1, 0.3
    duration = 26 # time in seconds
    skip_detect = 5 # skip detection every 5 frames
    desired_interval = 2 # proccess every 2 frames 

    if generate_frames: 
        generate_tracked_frames(video_files, folder_outs, duration, desired_interval, overlapping, skip_detect, max_age=1, min_hits=1, iou_threshold=0.3)

    if create_videos:
        vidcap = cv2.VideoCapture(video_file1)
        fps = vidcap.get(cv2.CAP_PROP_FPS)
        create_video(frames_dir=folder_out1,output_file= "dataset/Track-front.mp4",frame_size= (608,1080),framerate = fps//2)
        create_video(frames_dir=folder_out2,output_file= "dataset/Track-store.mp4",frame_size= (608,1080),framerate = fps//2)
