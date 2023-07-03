from IPython.display import HTML, display
from PIL import Image, ImageDraw, ImageFont
from base64 import b64encode
import seaborn as sns
import numpy as np
import os
import cv2
from tqdm.auto import tqdm
from sort import Sort
from centernet import ObjectDetection

def add_fifth_axis(array, default_value = 100):
    fifth_axis = np.full((array.shape[0], 1), default_value)
    result = np.hstack((array, fifth_axis))
    return result

def match_ID(overlapping,detections):
    """
    overlapping (np.array): the array of 4 cordination of the overlapping [x1,y1,x2,y2]
    detections (np.array): arraies with 5 col the first 4 is bounding box and the last is the id
    return (int): the id of the detected object with the most closet set of cordinations to the overlapping area
    """
    min_distance = np.inf
    overlapping_center = np.array([overlapping[0]+overlapping[2]/2,overlapping[1]+overlapping[3]/2])
    closest_id = 0

    for detection in detections:
        id = detection[-1]
        detection_center = np.array([detection[0]+detection[2]/2,detection[1]+detection[3]/2])
        distance = np.sqrt((detection_center[0] - overlapping_center[0])**2 + (detection_center[1] - overlapping_center[1])**2)
        if distance < min_distance:
            min_distance = distance
            closest_id = id
    return closest_id

class VisTrack:
    def __init__(self, unique_colors=400):
        """
        unique_colors (int): The number of unique colors (the number of unique colors dos not need to be greater than the max id)
        """
        self._unique_colors = unique_colors
        self._id_dict = {}
        self.p = np.zeros(unique_colors)
        self._colors = (np.array(sns.color_palette("hls", unique_colors))*255).astype(np.uint8)

    def _get_color(self, i):
        return tuple(self._colors[i])

    def _color(self, i):
        if i not in self._id_dict:
            inp = (self.p.max() - self.p ) + 1 
            if any(self.p == 0):
                nzidx = np.where(self.p != 0)[0]
                inp[nzidx] = 0
            soft_inp = inp / inp.sum()

            ic = np.random.choice(np.arange(self._unique_colors, dtype=int), p=soft_inp)
            self._id_dict[i] = ic

            self.p[ic] += 1

        ic = self._id_dict[i]
        return self._get_color(ic)

    def draw_bounding_boxes(self, im: Image, bboxes: np.ndarray, ids: np.ndarray) -> Image:
        """
        im (Image): The image 
        bboxes (np.ndarray): The bounding boxes. [[x1,y1,x2,y2],...]
        ids (np.ndarray): The id's for the bounding boxes
        scores (np.ndarray): The scores's for the bounding boxes
        """
        im = im.copy()
        draw = ImageDraw.Draw(im)

        for bbox, id_ in zip(bboxes, ids):
            color = self._color(id_)
            draw.rectangle((*bbox.astype(np.int64),), outline=color)

            text = f'{id_}'
            #text_w, text_h = draw.textsize(text)

            #increase text size
            font_size = 32
            font = ImageFont.truetype("Gidole-Regular.ttf", size=font_size)

            position = (bbox[0], bbox[1])
            bbox = draw.textbbox(position, text, font=font)
            draw.rectangle(bbox, fill=color, outline=color)
            draw.text(position, text, fill=(0, 0, 0), font=font)

        return im
class Camera:
    def __init__ (self, name, vidoePath, overlappingDic):
        self.name = name
        self.vidoePath = vidoePath
        self.overlappingDic = overlappingDic
        self.resDic = {}

    def create_res(self,duration,desired_interval,skip_detect,sort : Sort, obj : ObjectDetection):
        vidcap = cv2.VideoCapture(self.vidoePath)
        fps = vidcap.get(cv2.CAP_PROP_FPS)
        est_tot_frames = int(duration * fps)
        frame_count = 0
        print(f'creating res for {self.name}')
        for i in tqdm(range(0,est_tot_frames,desired_interval)):
            vidcap.set(cv2.CAP_PROP_POS_FRAMES, i)
            sucsses, frame = vidcap.read()
            if not sucsses:
                break

            # get boxes from object detction
            if (frame_count % skip_detect <= sort.min_hits):
                boxes, classes, scores = obj.predict(frame)
                detections_in_frame = len(boxes)
                if detections_in_frame:
                    # centernet will do detection on all the COCO classes. "person" is class number 0 
                    idxs = np.where(classes == 0)[0]
                    boxes = boxes[idxs]
                    scores = scores[idxs]
                else:
                    boxes = np.empty((0, 4))
            #get boxes from tracking
            else:
                if len(res):
                    boxes = res[:,:-1]
                else:
                    boxes = np.empty((0, 4))
                    
            dets = add_fifth_axis(boxes)
            res = sort.update(dets)
            self.resDic[frame_count] = res

            frame_count += 1
    
    def detect_prev_camera(self,res : np.ndarray):
        '''
        get the res of the bunding boxes and return the name of the camera that the object came from
        '''
        min_distance = np.inf
        camera = ""
        res_center = np.array([res[0]+res[2]/2,res[1]+res[3]/2])

        for key, value in self.overlappingDic.items():
            value_center = np.array([value[0]+value[2]/2,value[1]+value[3]/2])
            distance = np.sqrt((value_center[0] - res_center[0])**2 + (value_center[1] - res_center[1])**2)
            if distance < min_distance:
                min_distance = distance
                camera = key
        return camera
    
    def update_frame(self, frame, camerasDic, mapping_ids):
        '''
        update res[-1] to the correct id from mappings ID
        we will on each frame on it's res, from each res we will check if the id is already in mapping_ids
        if not, we will use detecet_prev_camera to identify from which camera did the obejct came. 
        then we will use macth_ID to get the id from the previous camera.
        we will update the id in mappings_ids and return the mappings_ids dic.
        '''
        for index, res in enumerate(self.resDic[frame]):
                id = int(res[-1])
                if id not in mapping_ids:
                    prev_camera = self.detect_prev_camera(res)
                    overlapping = camerasDic[prev_camera].overlappingDic[self.name]
                    detections = camerasDic[prev_camera].resDic[frame]
                    new_id = match_ID(overlapping,detections)
                    mapping_ids[id] = new_id
                self.resDic[frame][index,-1] = mapping_ids[id]
        return mapping_ids
    
    def draw_bounding_boxes(self, duration, desired_interval, vt, folder_out):
        '''
        this function will take the updated res and draw of each frame the corresponding res
        '''
        vidcap = cv2.VideoCapture(self.vidoePath)
        fps = vidcap.get(cv2.CAP_PROP_FPS)
        est_tot_frames = int(duration * fps)
        frame_count = 0
        print(f'drawing bounding boxes on {self.name} save it to {folder_out}')
        for i in tqdm(range(0,est_tot_frames,desired_interval)):
            vidcap.set(cv2.CAP_PROP_POS_FRAMES, i)
            sucsses, frame = vidcap.read()
            if not sucsses:
                break
            boxes_track = self.resDic[frame_count][:,:-1]
            boces_ids = self.resDic[frame_count][:,-1].astype(int)

            p_frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if len(self.resDic[frame_count]) > 0:
                p_frame = vt.draw_bounding_boxes(p_frame, boxes_track, boces_ids)
            p_frame.save(os.path.join(folder_out, f"{frame_count:03d}.png"))
            frame_count += 1

    def create_vidoe(self, output_file, frames_dir, desired_interval,frame_size = (1280,720), codec = "mp4v"):
        """
        frames_dir (str): The folder that has the tracked frames
        output_file (str): The file the video will be saved in 
        frame_size: The size of the frame (Width, Hight)
        coded: The type of the video
        """
        print(f'Creating video for {self.name} save to {output_file}')
        if os.path.exists(output_file):
            os.remove(output_file)

        frame_files = sorted(os.listdir(frames_dir))
        vidcap = cv2.VideoCapture(self.vidoePath)
        fps = vidcap.get(cv2.CAP_PROP_FPS)
        framerate = fps // desired_interval

        # Create a VideoWriter object
        output_vid = cv2.VideoWriter(output_file, cv2.VideoWriter_fourcc(*codec), framerate, frame_size)

        # Iterate through the frames and write them to the video
        for frame_file in frame_files:
            frame_path = os.path.join(frames_dir, frame_file)
            frame = cv2.imread(frame_path)
            output_vid.write(frame)

        # Release the VideoWriter object and close the video file
        output_vid.release()

    
class CameraFront(Camera):
    def __init__(self, name, vidoePath, overlappingDic):
        super().__init__(name, vidoePath, overlappingDic)
        self.counterID = 1

    def update_frame(self, frame, camerasDic, mapping_ids):
        '''
        update res[-1] to the correct id from mappings ID
        we will on each frame on it's res, from each res we will check if the id is already in mapping_ids
        if not, we will use detecet_prev_camera to identify from which camera did the obejct came. 
        then we will use macth_ID to get the id from the previous camera.
        we will update the id in mappings_ids and return the mappings_ids dic.
        '''
        for index, res in enumerate(self.resDic[frame]):
                id = int(res[-1])
                if id not in mapping_ids:
                    prev_camera = self.detect_prev_camera(res)
                    if prev_camera == 'door':
                        new_id = self.counterID
                        self.counterID += 1
                    else:
                        overlapping = camerasDic[prev_camera].overlappingDic[self.name]
                        detections = camerasDic[prev_camera].resDic[frame]
                        new_id = match_ID(overlapping,detections)
                    mapping_ids[id] = new_id
                self.resDic[frame][index,-1] = mapping_ids[id]
        return mapping_ids




