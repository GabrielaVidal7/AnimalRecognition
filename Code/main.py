import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer, util
from PIL import Image
# from IPython.display import Image as ShowImage
# from IPython.display import display
from os import listdir
from os.path import isfile, join
from sklearn.neighbors import NearestNeighbors

