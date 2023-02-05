import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer, util
from PIL import Image
# from IPython.display import Image as ShowImage
# from IPython.display import display
from os import listdir
from os.path import isfile, join
from sklearn.neighbors import NearestNeighbors
from zipfile import ZipFile
from sklearn.model_selection import train_test_split
from imblearn.under_sampling import RandomUnderSampler

from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier

data_url = 'Dataset\images_parte10.zip'


with ZipFile(data_url, 'r') as zip:
    # printing all the contents of the zip file
    zip.printdir()
  
    # extracting all the files
    print('Extracting all the files now...')
    zip.extractall()
    print('Done!')
