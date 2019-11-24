# This is a very basic image that just installs all the dependencies and runs the python script.
FROM ubuntu

RUN apt-get update
RUN apt-get install python3-pip -y

# Set the port on which the app runs; make both values the same.
#
# IMPORTANT: When deploying to Azure App Service, go to the App Service on the Azure
# portal, navigate to the Applications Settings blade, and create a setting named
# WEBSITES_PORT with a value that matches the port here (the Azure default is 80).
# You can also create a setting through the App Service Extension in VS Code.

EXPOSE 5000

WORKDIR /

COPY . .
ENV MODEL_PATH=/phishapp/files/phishing-model.h5

COPY requirements.txt /
RUN pip3 install --no-cache-dir -U pip
RUN pip3 install --no-cache-dir -r /requirements.txt

CMD ["python3", "startup.py"]
