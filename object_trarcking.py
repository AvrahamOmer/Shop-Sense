import os
import centernet

from sort import Sort
from lib import VisTrack, create_video, match_ID

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
    duration = 15 # time in seconds
    pbar = tqdm(total = int(fps * duration/2))
    sort = Sort(max_age=1, min_hits=3, iou_threshold=0.3)
    out_put_file = folder_out1 if video == video_file1 else folder_out2

    for i in range(int(fps * duration)):
        ret, frame = vidcap.read()
        if not ret:
            break
        # Takeing even frames
        if i%2 == 1:
            continue

        boxes, classes, scores = obj.predict(frame)
        detections_in_frame = len(boxes)
        if detections_in_frame:
            # centernet will do detection on all the COCO classes. "person" is class number 0 
            idxs = np.where(classes == 0)[0]
            boxes = boxes[idxs]
            scores = scores[idxs]
            classes = classes[idxs]
        else:
            boxes = np.empty((0, 5))
        

        dets = np.hstack((boxes, scores[:,np.newaxis]))
        res = sort.update(dets)

        if video == video_file1:
            dic_front[i] = res
        else:
            if len(res) != 0:
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
        

#Create a video
create_video(frames_dir=folder_out1,output_file= "dataset/Track-front.mp4",framerate = fps//2)
create_video(frames_dir=folder_out2,output_file= "dataset/Track-store.mp4",framerate = fps//2)

