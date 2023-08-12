from flask import Flask, jsonify
from object_trarcking import main as your_function

app = Flask(__name__)

@app.route('/api/get_data', methods=['GET'])
def api_route():
    result = your_function()  # Call your function
    return jsonify(result)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
