
# California Housing Data

# Population Block: 600-3000 people

# Buisiness Objective?

# How to use the data?

# Model metric? Business Metric?

#  Pipeline: A  data processing unit

# Current solutions? Currently Handled manually.
# Costly and time-consuming, poor estimates, off by 20%. 

# Use censu data

# Fram the problem: Is it supervised/unsupervised? Reinforment, regression/ classification. Batch learning/online learning

# Here: Supervised, (multiple) univariate regression problem.

# Performence Measure: RMSE, higher weightfor large errors
# RMSE vs MAE: l_1 vs l_2 norm: higher norm index focuses on large values . RMSE is sensitive to outliers.

# Assumptions:  predicting house prices can be classification
# "cheap", "medium", "expensive"

import os
import sys
import sklearn
import tarfile
import urllib.request
import numpy as np
import matplotlib.pyplot as plt

# Where to save the figures
PROJECT_ROOT_DIR = "."
CHAPTER_ID = "end_to_end_project"
IMAGES_PATH = os.path.join(PROJECT_ROOT_DIR, "images", CHAPTER_ID)
os.makedirs(IMAGES_PATH, exist_ok=True)



DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml2/master/"
HOUSING_PATH = os.path.join("datasets", "housing")
HOUSING_URL = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"

def fetch_housing_data(housing_url=HOUSING_URL, housing_path=HOUSING_PATH):
    if not os.path.isdir(housing_path):
        os.makedirs(housing_path)
    tgz_path = os.path.join(housing_path, "housing.tgz")
    urllib.request.urlretrieve(housing_url, tgz_path)
    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(path=housing_path)
    housing_tgz.close()

fetch_housing_data()
