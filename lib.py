from IPython.display import HTML, display
import PIL.Image
from base64 import b64encode
import seaborn as sns
import ffmpeg
import numpy as np
import os
import cv2

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
        id = detection[4]
        detection_center = np.array([detection[0]+detection[2]/2,detection[1]+detection[3]/2])
        distance = np.sqrt((detection_center[0] - overlapping_center[0])**2 + (detection_center[1] - overlapping_center[1])**2)
        if distance < min_distance:
            min_distance = distance
            closest_id = id
    return closest_id

def show_video(video_path, video_width = "fill"):
  """
  video_path (str): The path to the video
  video_width: Width for the window the video will be shown in 
  """
  video_file = open(video_path, "r+b").read()

  video_url = f"data:video/mp4;base64,{b64encode(video_file).decode()}"
  html_content = f"""<video width={video_width} controls><source src="{video_url}"></video>"""
  with open("index.html", "w") as file:
        file.write(html_content)

  print("index.html file created.")
def create_video(frames_dir = 'Track', output_file = 'movie.mp4', framerate = 25, frame_size = (1280,720), codec = "mp4v"):
  """
  frames_dir (str): The folder that has the tracked frames
  output_file (str): The file the video will be saved in 
  framerate (float): The framerate for the video
  frame_size: The size of the frame (Width, Hight)
  coded: The type of the video
  """
  if os.path.exists(output_file):
    os.remove(output_file)

  frame_files = sorted(os.listdir(frames_dir))

  # Create a VideoWriter object
  output_vid = cv2.VideoWriter(output_file, cv2.VideoWriter_fourcc(*codec), framerate, frame_size)

  # Iterate through the frames and write them to the video
  for frame_file in frame_files:
      frame_path = os.path.join(frames_dir, frame_file)
      frame = cv2.imread(frame_path)
      output_vid.write(frame)

  # Release the VideoWriter object and close the video file
  output_vid.release()


  

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

    def draw_bounding_boxes(self, im: PIL.Image, bboxes: np.ndarray, ids: np.ndarray,
                        scores: np.ndarray) -> PIL.Image:
        """
        im (PIL.Image): The image 
        bboxes (np.ndarray): The bounding boxes. [[x1,y1,x2,y2],...]
        ids (np.ndarray): The id's for the bounding boxes
        scores (np.ndarray): The scores's for the bounding boxes
        """
        im = im.copy()
        draw = PIL.ImageDraw.Draw(im)

        for bbox, id_, score in zip(bboxes, ids, scores):
            color = self._color(id_)
            draw.rectangle((*bbox.astype(np.int64),), outline=color)

            text = f'{id_}: {int(100 * score)}%'
            text_w, text_h = draw.textsize(text)
            draw.rectangle((bbox[0], bbox[1], bbox[0] + text_w, bbox[1] + text_h), fill=color, outline=color)
            draw.text((bbox[0], bbox[1]), text, fill=(0, 0, 0))

        return im
