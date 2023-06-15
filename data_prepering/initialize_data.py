"""extract images from videos"""
import cv2
import os
import shutil


def extract_images(path_in, path_out):
    """function to extract images from videos"""
    count = 0
    vidcap = cv2.VideoCapture(path_in)
    while True:
        success,image = vidcap.read()
        if not success:
            break
        cv2.imwrite( path_out + f'/frame{count}.jpg', image)     # save frame as JPEG file
        count = count + 1

if __name__=="__main__":

    folder_out1 = "./dataset/frames/front_frames"
    if os.path.exists(folder_out1):
        shutil.rmtree(folder_out1)
    os.makedirs(folder_out1)

    folder_out2 = "./dataset/frames/store_frames"
    if os.path.exists(folder_out2):
        shutil.rmtree(folder_out2)
    os.makedirs(folder_out2)

    input_paths = ["./dataset/videos/front_2.mp4","./dataset/videos/store_2.mp4"]
    output_dirs = ["./dataset/frames/front_frames","./dataset/frames/store_frames"]

    for i, input_path in enumerate(input_paths):
        extract_images(input_path, output_dirs[i])
        print("Extracted frames from video: ",input_path)
        print("Saved to directory: ",output_dirs[i])
        print(" ")
    