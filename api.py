from flask import Flask, jsonify , request
from object_trarcking import main
from lib import Camera, CameraFront

app = Flask(__name__)

@app.route('/api/get_data', methods=['POST'])
def api_route():
    data = request.get_json()
    cameras = data['cameras']
    front = cameras['front']
    store = cameras['store']
    camerasDic = {}
    camerasDic[front["name"]] = CameraFront(name=front["name"], vidoePath=front["path"], overlappingDic=front["overlapping"])
    for camera in store:
        camerasDic[camera["name"]] = Camera(name=camera["name"], vidoePath=camera["path"], overlappingDic=camera["overlapping"])
    result = main(camerasDic) 
    return jsonify(result)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
