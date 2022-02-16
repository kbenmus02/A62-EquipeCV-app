# 2021-11-24 20h43
from datetime import datetime 
from PIL import Image
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, confusion_matrix, classification_report, precision_recall_fscore_support
from sklearn.model_selection import train_test_split, learning_curve
from tqdm import tqdm


import cv2
import gc
import glob
import gzip #http://henrysmac.org/blog/2010/3/15/python-pickle-example-including-gzip-for-compression.html
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
import os
import pandas as pd
import pickle
import random
import scipy.sparse as sp
import seaborn as sns


