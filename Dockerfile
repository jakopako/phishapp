FROM tiangolo/uvicorn-gunicorn-fastapi:python3.7

COPY requirements.txt /
RUN pip3 install --no-cache-dir -r /requirements.txt

COPY ./phishapp /phishapp

ENV MODULE_NAME=phishapp.main
ENV MODEL_PATH=/phishapp/files/phishing-model/phishing-model.h5
ENV LOGO_PATH=/phishapp/files/logos
