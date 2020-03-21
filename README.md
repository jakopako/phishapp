# Phishapp

This entire project is an attempt to classify screenshots of websites according to the brand they
represent/target and make this functionality available through a REST API. 

Currently all this is, is a neural network that has been trained on a few thousand
images of fake login pages (or generally pages that phish for sensitive data). The quality of the
classification is not very good (yet :D) but will hopefully become better in the future. Some ideas
include extending the image set, removing strange artifacts from the current image set (the top of
the images currently have a black bar containing the url corresponding to the screenshot..),
redesigning the layout of the neural network itself, etc. The current version can be tested at
http://phishapp.dhondtdoit.ch 

There are still many, many things to improve so don't expect too much. In fact, the classification
works rather poorly on images that are not very close to the training set.

## Training the Model

In the package `phishapp.phishmodel` there is a `Detector` class that is used for both training
a model and predicting based on a trained model. For details have a look at the description in
the code. The script containing the `Detector` class can be invoked directly from the commandline
to train the model. The only thing that is needed is a directory with the following structure:
```
dir/
  -- train/
    -- class_one/
    -- class_two/
  -- validation/
    -- class_one/
    -- class_two/
```


## Building and pushing the container

### Azure
Build the container locally. Tag it according to where you want to publish it. Execute
the following command in the base directory of this repository, where the `Dockerfile` is
located.
```
$ docker build -t phishapp.azurecr.io/phishapp:latest .
```
Push the image to a container registry.
```
$ docker push phishapp.azurecr.io/phishapp
```

### Heroku
Do
 ```
$ heroku container:login
```
then
```
$ heroku container:push web -a phishapp
```
then
```
$ heroku container:release web -a phishapp
```

## Testing
There are a few test images located in the `example_images` directory. You will see that not all
of them are classified correctly. Note, that none of them have been used for training, however, 
`paypal3.png` is very "close" to the majority of Paypal images that have been used for training
whereas the other two images are less similar to the training set.
### Locally
**REST API:**
There are two different ways of testing the API locally. You can either invoke the python script
`startup.py` directly or you can first build the docker container and then run it locally. For
quick testing of the code the former obviously makes more sense. To post an image to the API
you can use the `phishmodel/test_app.py` script.

**Detector:**
You can directly test the prediction method by invoking the script that contains the `Detector`
class with the according parameters.

### Remotely
To test the publicly available API you can use the same script as above (`phishmodel/test_app.py`)
but use `http://phishapp.dhondtdoit.ch/predict` as endpoint. In `phishapp/` do:
```
(phishapp)$ python test_app.py -u http://phishapp.dhondtdoit.ch/predict -p ../example_images/paypal3.png
```
Output:
```
{
    "Amazon": "6.2584877e-07",
    "Apple": "1.385808e-05",
    "Microsoft Office365": "0.0",
    "Netflix": "2.4437904e-06",
    "NotActive": "1.9669533e-06",
    "Paypal": "0.8580924",
    "Postfinance": "0.0",
    "Swisscom": "1.4901161e-07",
    "UBS": "5.17066e-07"
}
```

