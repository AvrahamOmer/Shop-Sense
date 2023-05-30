"""extract images from videos"""
import cv2

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
    input_paths = ["./dataset/videos/front_1.mp4","./dataset/videos/store_1.mp4"]
    output_dirs = ["./dataset/frames/front_frames","./dataset/frames/store_frames"]

    for i, _ in enumerate(input_paths):
        extract_images(input_paths[i], output_dirs[i])
        print("Extracted frames from video: ",input_paths[i])
        print("Saved to directory: ",output_dirs[i])
        print(" ")
    