from flask import Flask, make_response, jsonify, request, abort
import os
from phishapp.phishmodel import Detector

app = Flask(__name__)

detector = Detector()
try:
    model_path = os.environ['MODEL_PATH']
    detector.load_model(model_path)
except KeyError:
    detector.load_model('phishapp/files/phishing-model.h5')


@app.route('/predict', methods=['POST'])
def predict():
    if not request.json or 'image' not in request.json:
        abort(400)
    image_base64 = request.json['image']
    image = detector.preprocess_image_from_base64(image_base64)
    prediction = detector.predict(image)
    prediction = dict([(k, str(v)) for k, v in prediction.items()])
    return jsonify(prediction), 200


@app.errorhandler(404)
def not_found(error):
    return make_response(jsonify({'error': 'Not found'}), 404)


if __name__ == '__main__':
    app.run(debug=True)
