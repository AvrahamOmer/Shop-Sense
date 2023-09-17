from flask import Flask, request, send_file
from flask_cors import CORS
from object_trarcking import main
from lib import Camera, CameraFront
import json
from werkzeug.utils import secure_filename
import os
from moviepy.editor import ImageSequenceClip
import zipfile
import shutil

app = Flask(__name__)
CORS(app)

def remove_files_and_folders(directory_path):
    for filename in os.listdir(directory_path):
        file_path = os.path.join(directory_path, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print(f'Failed to delete {file_path}. Reason: {e}')


def create_video(image_folder, fps):
    clip = ImageSequenceClip(image_folder, fps=fps)
    clip.write_videofile("output.mp4")
    clip.close()

def string_to_array(s):
    return [int(x) for x in s.split(',')]

def convert_to_front_store(files,datas):
    front = {}
    store = []

    zipped = zip(files,datas)

    for i, item in enumerate(zipped):
        file_path = item[0]
        video_name = item[1]["name"]
        overlapping = {}
        for overLappingName, overlappingStr in item[1]["overlapping"].items():
            overlapping[overLappingName] = string_to_array(overlappingStr)

        if i == 0:
            front["name"] = video_name
            front["path"] = file_path
            front["overlapping"] = overlapping
        else:
            store.append({
                "name": video_name,
                "path": file_path,
                "overlapping": overlapping
            })
    return front, store


@app.route('/api/get_data', methods=['POST'])
def api_route():
    files_array = []
    for _, file in request.files.items():
        filename = os.path.join('share', secure_filename(file.filename))
        file.save(filename)
        files_array.append(filename)

    data_array = []
    for key in request.form.keys():
        if key.startswith('data'):
            data = request.form.get(key)
            data_dict = json.loads(data)
            data_array.append(data_dict)

    print(f'files_array: {files_array}')
    print(f'data_array: {data_array}')

    front, store = convert_to_front_store(files_array, data_array)

    camerasDic = {}
    camerasDic[front["name"]] = CameraFront(name=front["name"], vidoePath=front["path"], overlappingDic=front["overlapping"])
    print(f'front: {front["name"]}, {front["path"]}, {front["overlapping"]}''')
    for camera in store:
        camerasDic[camera["name"]] = Camera(name=camera["name"], vidoePath=camera["path"], overlappingDic=camera["overlapping"])
        print(f'store: {camera["name"]}, {camera["path"]}, {camera["overlapping"]}''')
    files, time_per_id, total_customers, avg_time, people_per_second = main(camerasDic)

    res = {
        "time_per_id": time_per_id, 
        "total_customers": total_customers,
        "avg_time": avg_time,
        "people_per_second": people_per_second
    }

    # Specify the file path for the new JSON file
    file_path = "final_results.json"

    # Step 1: Create a new JSON file and write the 'res' dictionary to it
    with open(file_path, 'w') as file:
        json.dump(res, file)

    #Step 2: add json file to our zip file
    files.append(file_path)


    zip_filename = 'files.zip'
    with zipfile.ZipFile(zip_filename, 'w') as zip_file:
        for file in files:
            zip_file.write(file)
    
    return send_file(zip_filename, mimetype='zip')



if __name__ == '__main__':
    remove_files_and_folders('share')
    remove_files_and_folders('Track')
    remove_files_and_folders('dataset/marked')
    app.run(host='0.0.0.0', port=5000)
