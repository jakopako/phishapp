  # This is a very basic image that just installs all the dependencies and runs the python script.
FROM ubuntu:latest

RUN apt-get update
RUN apt-get install python3-pip -y
RUN DEBIAN_FRONTEND=noninteractive apt-get install libgl1-mesa-glx libglib2.0-0 -y

# Set the port on which the app runs; make both values the same.
#
# IMPORTANT: When deploying to Azure App Service, go to the App Service on the Azure
# portal, navigate to the Applications Settings blade, and create a setting named
# WEBSITES_PORT with a value that matches the port here (the Azure default is 80).
# You can also create a setting through the App Service Extension in VS Code.

COPY requirements.txt /
RUN pip3 install --no-cache-dir -U pip
RUN pip3 install --no-cache-dir -r /requirements.txt

COPY ./phishapp /phishapp

ENV MODEL_PATH=/phishapp/files/phishing-model/phishing-model.h5
ENV LOGO_PATH=/phishapp/files/logos
ENV PORT=5000

CMD uvicorn phishapp.main:app --host=0.0.0.0 --port=$PORT
