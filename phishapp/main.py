import os
from phishapp.detector.logodetector import LogoDetector
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn

app = FastAPI()

l_detect = LogoDetector()
if 'LOGO_PATH' in os.environ:
    logo_path = os.environ['LOGO_PATH']
    l_detect.load_logos(logo_path)
else:
    l_detect.load_logos('files/logos')

screenshots = False
if 'ENABLE_SCREENSHOTS' in os.environ:
    from phishapp.detector.screenshotdetector import ScreenshotDetector
    screenshots = True
    s_detect = ScreenshotDetector()
    if 'MODEL_PATH' in os.environ:
        model_path = os.environ['MODEL_PATH']
        s_detect.load_model(model_path)
    else:
        s_detect.load_model('phishapp/files/phishing-model/phishing-model.h5')


class Image(BaseModel):
    base64_data: str


@app.post('/screenshot/predict')
def predict(image: Image):
    if screenshots:
        image = s_detect.preprocess_image_from_base64(image.base64_data)
        prediction = s_detect.predict(image)
        prediction = dict([(k, str(v)) for k, v in prediction.items()])
        return prediction
    else:
        return {"error": "This function is currently disabled."}


@app.post('/logo/detect')
def detect_logo(image: Image):
    try:
        image = l_detect.preprocess_image_from_base64(image.base64_data)
    except Exception as e:
        return {"error": str(e)}
    logo_positions = l_detect.find_logos(image)
    return logo_positions


@app.get('/logo/list')
def list_logos():
    return l_detect.get_all_supported_brands()


if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    uvicorn.run(app, host='0.0.0.0', port=port)
