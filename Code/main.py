import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer, util
from PIL import Image
# from IPython.display import Image as ShowImage
# from IPython.display import display
from os import listdir
from os.path import isfile, join
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import train_test_split
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
# import ktrain
# from ktrain import vision as vis
import os
from shutil import copyfile

def get_images (path, subpath):
    images = [subpath+'/'+f for f in listdir(path+subpath) if isfile(join(path+subpath, f))]
    len(images)
    return images

def make_dir(dirPath):
    if not os.path.isdir(dirPath):
        os.mkdir(dirPath)
    return

path = 'Dataset/'

mypath = ['cheetah', 'hyena']
images=[]
for subpath in mypath:
  images+=get_images(path, subpath)
# print(len(images))


image_feature_extractor = SentenceTransformer('clip-ViT-B-32')
df = pd.DataFrame(images)
df.columns = ['filename']

X = []
for index,row in df.iterrows():
  img_features = image_feature_extractor.encode(Image.open(path+row['filename']))
  X.append(img_features)
df['features'] = X

df.loc[df['filename'].str.contains('cheetah'),'class'] = 0
df.loc[df['filename'].str.contains('hyena'),'class'] = 1


print(df)

# Split train and test
Y = df['class']

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=42)

# target distribution
(unique, counts) = np.unique(y_train, return_counts=True)
np.asarray((unique, counts)).T


# ===================================================
# KNN Method k=[1, 3, 5, 7, 11]
# ===================================================
K = [1, 3, 5, 7, 11]

Acc = pd.DataFrame(K)
Acc.columns = ['K']
acc_knn=[]

for k in K:
    knn = KNeighborsClassifier(n_neighbors=k,metric='cosine')
    knn.fit(X_train,y_train)
    # Results
    acc_knn.append(knn.score(X_test, y_test))
    #   print('k: ', k, '\tacurácia = %.5f' %acc)


# ===================================================
# Neural Network Method
# ===================================================
'''make_dir('data/train')
make_dir('data/test')       #Creating directories for train and test

x_train ,x_test = train_test_split(images,test_size=0.3)    #Split 30% test, 70% train

for img in x_train:
    img_class = img.split('/')[0]

    if not os.path.exists('data/train/'+img_class):
        os.makedirs('data/train/'+img_class)

    copyfile(path+img, 'data/train/'+img)

for img in x_test:
    img_class = img.split('_')[0]

    if not os.path.exists('data/test/'+img_class):
        os.makedirs('data/test/'+img_class)

    copyfile(path+img, 'data/test/'+img)

(train_data, val_data, preproc) = vis.images_from_folder('data/')
vis.print_image_classifiers()

model = vis.image_classifier('pretrained_resnet50', train_data, val_data)
learner = ktrain.get_learner(model=model, train_data=train_data, val_data=val_data, 
                             workers=8, use_multiprocessing=False, batch_size=64)
learner.fit_onecycle(1e-4, 1)
predictor = ktrain.get_predictor(learner.model, preproc)
acc_nn = learner.validate(val_data, predictor, class_names = predictor.get_classes())

print(acc_nn)'''
# for acc in acc_knn:
#     print(acc)
