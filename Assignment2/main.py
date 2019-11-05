import pandas as pd
import urllib
from sklearn.model_selection import train_test_split
import os
import numpy as np
from operator import itemgetter
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
from data.data import *
from LinReg.Model import *

# requirements :
#     recommended platform : ubuntu
#     python == 3.7
#     pip install pandas
#     pip install numpy
#     pip install sklearn
#     pip install seaborn
#     pip install matplotlib


if __name__ == '__main__':
    loadData()
    data = readpickle('data.pkl')

    linearReg(data)