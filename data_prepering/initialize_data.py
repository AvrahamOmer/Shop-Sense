import cv2

def extractImages(pathIn, pathOut):
    count = 0
    vidcap = cv2.VideoCapture(pathIn)
    while True:
        success,image = vidcap.read()
        if not success:
            break
        cv2.imwrite( pathOut + "/frame%d.jpg" % count, image)     # save frame as JPEG file
        count = count + 1

if __name__=="__main__":
    input_paths = ["./dataset/videos/front_1.mp4","./dataset/videos/store_1.mp4"]
    output_dirs = ["./dataset/frames/front_frames","./dataset/frames/store_frames"]

    for i in range(len(input_paths)):
        extractImages(input_paths[i], output_dirs[i])
        print("Extracted frames from video: ",input_paths[i])
        print("Saved to directory: ",output_dirs[i])
        print(" ")