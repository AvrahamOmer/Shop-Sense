from flask import Flask, jsonify , request, send_file
from flask_cors import CORS
from object_trarcking import main
from lib import Camera, CameraFront
import json
from werkzeug.utils import secure_filename
import os
import imageio
from moviepy.editor import ImageSequenceClip
import zipfile

app = Flask(__name__)
CORS(app)

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
    files = main(camerasDic)


    zip_filename = 'files.zip'
    with zipfile.ZipFile(zip_filename, 'w') as zip_file:
        for file in files:
            zip_file.write(file)
    
    return send_file(zip_filename, mimetype='zip')



if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
