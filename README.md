# Conv-Nets-for-Multi-label-classification

[The paper](http://cs231n.stanford.edu/reports/2017/pdfs/908.pdf)

[The dataset](https://www.kaggle.com/c/planet-understanding-the-amazon-from-space/data)

# Datasets:

All the images in train-jpg folder were used to train the models. The images in test-jpg and test-jpg-additional folders were used to test the accuracy of the model.

# Machine Used:
 The group has executed this program on Digital Ocean droplets with the following configurations<br />
 64GB RAM, 32 vCPUs, 400GB SSD Disk<br />
 

# How To Run:

1. Clone the Repository

2. Install python3 and install all these python packages using pip-8.1.1:<br />
  -numpy-1.14.2<br /> 
  -opencv-python-3.4.0.12<br /> 
  -pandas-0.22.0<br />
  -python-dateutil-2.7.2<br /> 
  -pytz-2018.4<br />
  -h5py-2.7.1<br /> 
  -scikit-learn-0.19.1<br /> 
  -sklearn-0.0<br />
  -scipy-1.0.1<br />
  -keras-2.1.5<br />
  -absl-py-0.2.0<br />
  -astor-0.6.2<br />
  -bleach-1.5.0<br />
  -gast-0.2.0<br />
  -grpcio-1.11.0<br />
  -html5lib-0.9999999<br /> 
  -markdown-2.6.11<br />
  -protobuf-3.5.2.post1<br /> 
  -tensorboard-1.7.0<br />
  -tensorflow-1.7.0<br />
  -termcolor-1.1.0<br />
  -werkzeug-0.14.1<br />
 

3. Run files:<br />
  -`python3 Amazon.py` to get Baseline model predictions<br />
  -`python3 VGG-16.py` to get VGG-16 model predictions<br />
  -`python3 Inception_mod.py` to get Inception model predictions<br />
  -`python3 Resnet50.py` to get ResNet-50 model predictions<br />
  
Use the csv file created by running these commands in this [kaggle website](https://www.kaggle.com/c/planet-understanding-the-amazon-from-space/leaderboard) under late submissions to get the accuracy of the model.
Note: All the python files make submission_keras.csv file which needs to be submitted to the above Kaggle link.
