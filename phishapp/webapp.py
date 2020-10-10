from flask import Flask, make_response, jsonify, request, abort
import os
from phishapp.detector import ScreenshotDetector
from phishapp.detector import LogoDetector

app = Flask(__name__)

s_detect = ScreenshotDetector()
try:
    model_path = os.environ['MODEL_PATH']
    s_detect.load_model(model_path)
except KeyError:
    s_detect.load_model('phishapp/files/phishing-model.h5')

l_detect = LogoDetector()
try:
    logo_path = os.environ['LOGO_PATH']
    l_detect.load_logos(logo_path)
except KeyError:
    l_detect.load_logos('logos')


@app.route('/predict', methods=['POST'])
def predict():
    if not request.json or 'image' not in request.json:
        abort(400)
    image_base64 = request.json['image']
    image = s_detect.preprocess_image_from_base64(image_base64)
    prediction = s_detect.predict(image)
    prediction = dict([(k, str(v)) for k, v in prediction.items()])
    return jsonify(prediction), 200


@app.route('/detect-logo', methods=['POST'])
def detect_logo():
    if not request.json or 'image' not in request.json:
        abort(400)
    image_base64 = request.json['image']
    image = l_detect.preprocess_image_from_base64(image_base64)
    logo_positions = l_detect.find_logos(image)
    return jsonify(logo_positions), 200


@app.errorhandler(404)
def not_found(error):
    return make_response(jsonify({'error': 'Not found'}), 404)


if __name__ == '__main__':
    app.run(debug=True)
