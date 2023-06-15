import os
import centernet
import time


from sort import Sort
from lib import VisTrack, create_video, match_ID, add_fifth_axis

import numpy as np
import PIL.Image
import cv2
from tqdm.auto import tqdm
import shutil

video_file1 = "./dataset/videos/front_1.mp4"
video_file2 = "./dataset/videos/store_1.mp4"
video_files = [video_file1, video_file2]
overlapping = np.array([375,360,500,700])
vt = VisTrack()
dic_front = {}
mapping_ids = {}
max_age, min_hits, iou_threshold = 1, 1, 0.3
sort = Sort(max_age, min_hits, iou_threshold)
duration = 15 # time in seconds
skip_detect = 5 # skip detection every 10 frames
id_object = 1 

# Default: num_classes=80
obj = centernet.ObjectDetection(num_classes=80)

# num_classes=80 and weights_path=None: Pre-trained COCO model will be loaded.
obj.load_weights(weights_path=None)


folder_out1 = "Track/Track-front"
if os.path.exists(folder_out1):
    shutil.rmtree(folder_out1)
os.makedirs(folder_out1)

folder_out2 = "Track/Track-store"
if os.path.exists(folder_out2):
    shutil.rmtree(folder_out2)
os.makedirs(folder_out2)


for video in video_files:
    vidcap = cv2.VideoCapture(video)
    fps = vidcap.get(cv2.CAP_PROP_FPS)
    est_tot_frames = int(duration * fps)  # Sets an upper bound # of frames in video clip
    n = 2                             # Desired interval of frames to include
    frame_count = 0

    pbar = tqdm(total = int(est_tot_frames // n))
    out_put_file = folder_out1 if video == video_file1 else folder_out2
    start_time = time.time()  # Record the start time
    for i in range(0,est_tot_frames,n):
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

        

#Create a video
create_video(frames_dir=folder_out1,output_file= "dataset/Track-front.mp4",framerate = fps//2)
create_video(frames_dir=folder_out2,output_file= "dataset/Track-store.mp4",framerate = fps//2)
