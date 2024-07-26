#!/usr/bin/env python
# coding: utf-8

# In[2]:


get_ipython().run_line_magic('cd', '../')


# In[23]:


import tensorflow as tf
print(tf.__version__)
import torch
print(f"Setup complete. Using torch {torch.__version__} ({torch.cuda.get_device_properties(0).name if torch.cuda.is_available() else 'CPU'})")

# Necessary/extra dependencies.
import os
import gc
import cv2
import glob
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
from shutil import copyfile
import matplotlib.pyplot as plt
import yaml
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
import os.path
from os import path
'''
import pydicom
from pydicom.pixel_data_handlers.util import apply_voi_lut
'''


# In[3]:


# Login 
import wandb
wandb.login()


# In[4]:


IMG_SIZE = 256
BATCH_SIZE = 16
EPOCHS = 35
USE_FOLD = False


# # Merge CSV files

# In[13]:


# Load csv file
df_image = pd.read_csv('../covid/train_image_level.csv')
df_label = pd.read_csv('../covid/train_study_level.csv')
df_meta = pd.read_csv('../covid/meta.csv')

# Modify values in the id column
df_image['id'] = df_image.apply(lambda row: row.id.split('_')[0], axis=1)
# Add absolute path
df_image['path'] = df_image.apply(lambda row: '../covid/train/'+row.id+'.jpg', axis=1)
# Get image level labels
df_image['image_level'] = df_image.apply(lambda row: row.label.split(' ')[0], axis=1)

df_image.head(5)


# In[14]:


#print(df_label.head(5))
df_label['StudyInstanceUID'] = df_label.apply(lambda row: row.id.split('_')[0], axis=1)
df_label = df_label.drop('id', axis=1)
def study_level(row):
    if row[0] == 1:
        return 'Negative'
    elif row[1] == 1:
        return 'Typical'
    elif row[2] == 1:
        return 'Indeterminate'
    elif row[3] == 1:
        return 'Atypical'
df_label['study_level'] = df_label.apply(lambda row: study_level(row), axis=1)
#print(df_label.head(5))

df = df_label.merge(df_image, on ='StudyInstanceUID', how='left')
df.head(5)


# In[15]:


# Original dimensions are required to scale the bounding box coordinates appropriately.
train_meta_df = df_meta.loc[df_meta.split == 'train']
test_meta_df = df_meta.loc[df_meta.split == 'test']
train_meta_df = train_meta_df.drop('split', axis=1)
train_meta_df.columns = ['id', 'dim0', 'dim1']

train_meta_df.head(2)


# In[16]:


# Merge both the dataframes
df = df.merge(train_meta_df, on='id',how="left")
df.to_csv('total.csv', index=False)
df.head(2)


# ## Clean wrong label image and use K-fold split train and valid

# In[21]:


if 'train_fold.csv' in os.listdir(os.getcwd()) and USE_FOLD:
    df_n = pd.read_csv('train_fold.csv')
else:
    df_total = pd.read_csv('total.csv')
    df_n = df_total
    
    # clear the date that has be labeled non-negative but no bbox
    for i in range(len(df_total)):
        #print(row)
        row = df_total.loc[df_total.index.values[i]]
        label = row.study_level
        box = row.boxes

        if label != 'Negative' and box is np.nan:
            df_n = df_n.drop(df_total.index.values[i])
            #df = df.drop(grp_df.index.values[i])
            #print(len(df_total))
    
    print(len(df_total), len(df_n))
    df_n = df_n.drop('boxes', axis=1).reset_index()
    print(df_n.head(5))
    
    # Create label for kfold
    Fold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for n, (train_index, val_index) in enumerate(Fold.split(df_n, df_n['study_level'])):
        df_n.loc[val_index, 'fold'] = int(n)
    df_n['fold'] = df_n['fold'].astype(int)
    
    df_n.to_csv('train_fold.csv', index=False)

df_n.head()


# In[22]:


# Create train and validation split with K-fold
if USE_FOLD:
    pass
else:
    # Remove existing dirs
    for fold in range(5):
        # Prepare train and valid df
        train_df = df_n.loc[df_n.fold != fold].reset_index(drop=True)
        valid_df = df_n.loc[df_n.fold == fold].reset_index(drop=True)
        
        try:
            shutil.rmtree(f'dataset_folds_{fold}/images')
            shutil.rmtree(f'dataset_folds_{fold}/labels')
        except:
            print('No dirs')

        # Make new dirs
        os.makedirs(f'dataset_folds_{fold}/images/train', exist_ok=True)
        os.makedirs(f'dataset_folds_{fold}/images/valid', exist_ok=True)
        os.makedirs(f'dataset_folds_{fold}/labels/train', exist_ok=True)
        os.makedirs(f'dataset_folds_{fold}/labels/valid', exist_ok=True)

        # Move the images to relevant split folder.
        for i in tqdm(range(len(train_df))):
            row = train_df.loc[i]
            copyfile(row.path, f'dataset_folds_{fold}/images/train/{row.id}.jpg')
            
        for i in tqdm(range(len(valid_df))):
            row = valid_df.loc[i]
            copyfile(row.path, f'dataset_folds_{fold}/images/valid/{row.id}.jpg')


# In[24]:


# Create 5 .yaml file for each fold

for f in range(5):
    data_yaml = dict(
        train = f'../dataset_folds_{f}/images/train',
        val = f'../dataset_folds_{f}/images/valid',
        nc = 4,
        names = ['Negative', 'Typical', 'Indeterminate', 'Atypical']
    )

    # Note that I am creating the file in the yolov5/data/ directory.
    with open(f'yolov5/data/data_fold_{f}.yaml', 'w') as outfile:
        yaml.dump(data_yaml, outfile, default_flow_style=True)
    
get_ipython().run_line_magic('cat', 'yolov5/data/data_fold_0.yaml')


# In[ ]:


hyp_yaml = dict(
    
    anchor_t= 4.0,
    box= 0.05,
    cls= 0.5,
    cls_pw= 1.0,
    copy_paste= 0.0,
    degrees= 0.0,
    fl_gamma= 0.0,
    fliplr= 0.5,
    flipud= 0.0,
    hsv_h= 0.015,
    hsv_s= 0.7,
    hsv_v= 0.4,
    iou_t= 0.2,
    lr0= 0.00001,
    lrf= 0.0002,
    mixup= 0.0,
    momentum= 0.937,
    mosaic= 1.0,
    obj= 1.0,
    obj_pw= 1.0,
    perspective= 0.0,
    scale= 0.5,
    shear= 0.0,
    translate= 0.1,
    warmup_bias_lr= 0.1,
    warmup_epochs= 3.0,
    warmup_momentum= 0.8,
    weight_decay= 0.0005,
)

# creating the yaml file in the yolov5/data/ directory.
with open('yolov5/data/hyp.yaml', 'w') as outfile:
    yaml.dump(hyp_yaml, outfile, default_flow_style=True)
    
get_ipython().run_line_magic('cat', 'yolov5/data/hyp.yaml')


# In[25]:


def convert(size, box):
    # Scaling factor
    dw = 1./(size[1]) 
    dh = 1./(size[0])
    
    # Convert the bbox into YOLO format 
    x = (box[0] + box[2])/2.0   # (xmin + xmax)/2
    y = (box[1] + box[3])/2.0   # (ymin + ymax)/2
    w = box[2] - box[0]         # xmax - xmin
    h = box[3] - box[1]         # ymax - ymin
    
    x = x*dw
    w = w*dw  
    y = y*dh 
    h = h*dh
    
    return [x,y,w,h]   # x_center y_center width height


# In[ ]:


def write_bbox_files(tmp_df, fold_num, split):
    path = f'dataset_folds_{fold}/labels/{split}'

    # make all the image has its own txt file
    for i in tqdm(range(len(df_n))):
        row = df_n.loc[i]
        

        filename = f'{path}/{row.id}.txt'

        # Scaling factor
        Scale_x = 256 / row.dim1
        Scale_y = 256 / row.dim0

        if row.image_level != 'Negative':
            bbox = []
            box = []
            for i, val in enumerate(row.label.split(' ')):
                if (i % 6 <= 1):
                    continue
                box.append(float(val))
                if i % 6 == 5:
                    box[0] = np.round(box[0]*Scale_x, 5)
                    box[1] = np.round(box[1]*Scale_y, 5)
                    box[2] = np.round(box[2]*Scale_x, 5)
                    box[3] = np.round(box[3]*Scale_y, 5)
                    bbox.append(convert([256,256],box))
                    box = [] 
            #print(bbox)

            with open(filename, 'w') as file:
                for box in bbox:
                    box = [1]+box
                    box = [str(i) for i in box]
                    box = ' '.join(box)
                    #print(box)
                    file.write(box)
                    file.write('\n')

if USE_FOLD:
    pass
else:
    # Prepare the txt files for bounding box
    for fold in range(5):
        # Prepare train and valid df
        train_df = df_n.loc[df_n.fold != fold].reset_index(drop=True)
        valid_df = df_n.loc[df_n.fold == fold].reset_index(drop=True)
        
        # prepare label for train
        write_bbox_files(train_df, fold, 'train')
        # prepare label for valid
        write_bbox_files(valid_df, fold, 'valid')


# # Train YOLOv5 

# In[ ]:


get_ipython().run_line_magic('cd', 'yolov5/')


# In[ ]:


for fold in range(5):    
    print('FOLD NUMBER: ', fold)
    get_ipython().system('python train.py --img {IMG_SIZE}                      --batch {BATCH_SIZE}                      --epochs {35}                      --data data_fold_{fold}.yaml                      --weights yolov5s.pt                      --optimizer SGD                     --save-period 1                     --project yolov5-folds                     --name yolov5s-fold-{fold}')
    print('###########################################################################################\n')


# # Test YOLOv5

# In[ ]:


TEST_PATH = f'/kaggle/tmp/test/'
def prepare_test_images():

    os.makedirs(TEST_PATH, exist_ok=True)

    for dirname, _, filenames in tqdm(os.walk(f'../covid/test')):
        for file in filenames:
            copyfile(os.path.join(dirname, file),os.path.join(TEST_PATH, file))
            
    


# In[ ]:


prepare_test_images()
print(f'Number of test images: {len(os.listdir(TEST_PATH))}')
lens = len(os.listdir(TEST_PATH))


# In[ ]:


# YOLO_MODEL_PATH = '../input/yolo-models/yolov5s-e-100-img-512.pt'
MODEL_WEIGHTS = [
    'yolov5-folds/yolov5s-fold-0/weights/best.pt',
    'yolov5-folds/yolov5s-fold-1/weights/best.pt',
    'yolov5-folds/yolov5s-fold-2/weights/best.pt',
    'yolov5-folds/yolov5s-fold-3/weights/best.pt',
    'yolov5-folds/yolov5s-fold-4/weights/best.pt',
]


CONFIDENCE = [
    0.326, 0.28, 0.277, 0.252, 0.308
]

get_ipython().system('python detect.py --weights {YOLO_MODEL_PATHS[0]} {YOLO_MODEL_PATHS[1]} {YOLO_MODEL_PATHS[2]} {YOLO_MODEL_PATHS[3]} {YOLO_MODEL_PATHS[4]}                                       --source {TEST_PATH}                                       --img {IMG_SIZE}                                       --conf 0.2                                       --iou-thres 0.5                                       --max-det 10                                       --save-txt                                       --save-conf                                      --nosave#')


# In[ ]:


PRED_PATH = 'runs/detect/exp/labels'
prediction_files = os.listdir(PRED_PATH)
print(f'Number of opacity predicted by YOLOv5: {len(prediction_files)}')


# # See the result

# In[ ]:


print(len(test_meta_df), lens)
test_meta_df.head(4)


# In[ ]:


def scale_wandb_format(bboxes):
    wandb_bboxes = []
    for b in bboxes:
        xc, yc = int(np.round(b[0]*IMG_SIZE)), int(np.round(b[1]*IMG_SIZE))
        w, h = int(np.round(b[2]*IMG_SIZE)), int(np.round(b[3]*IMG_SIZE))

        xmin = xc - int(np.round(w/2))
        ymin = yc - int(np.round(h/2))
        xmax = xc + int(np.round(w/2))
        ymax = yc + int(np.round(h/2))
        
        wandb_bboxes.append([xmin, ymin, xmax, ymax])
        
    return wandb_bboxes


# In[ ]:


def get_pred(file_path):
    confidence = []
    bboxes = []
    with open(file_path, 'r') as file:
        for line in file:
            preds = line.strip('\n').split(' ')
            preds = list(map(float, preds))
            confidence.append(preds[-1])
            bboxes.append(preds[1:-1])
            pre_class = int(preds[0])
        #pre_class.append(clas)
            
    return confidence, bboxes, pre_class


# In[ ]:


con,bbox,clas = get_pred('runs/detect/exp3/labels/fe501aa91e43.txt')
print(type(clas))
get_ipython().system("cat 'runs/detect/exp3/labels/fe501aa91e43.txt'")


# In[ ]:


bbox = scale_wandb_format(bbox)
print(bbox)


# In[ ]:


class_label_to_id = {
    'Negative for Pneumonia': 0,
    'Typical Appearance': 1,
    'Indeterminate Appearance': 2,
    'Atypical Appearance': 3
}

class_id_to_label = {val: key for key, val in class_label_to_id.items()}


# In[ ]:


def wandb_bbox(image, bboxes, label):
    all_boxes = []
    for bbox in bboxes:
        box_data = {"position": {
                        "minX": bbox[0],
                        "minY": bbox[1],
                        "maxX": bbox[2],
                        "maxY": bbox[3]
                    },
                     "class_id" : int(label),
                     "box_caption": class_id_to_label[label],
                     "domain" : "pixel"}
        all_boxes.append(box_data)
    

    return wandb.Image(image, boxes={
        "ground_truth": {
            "box_data": all_boxes,
          "class_labels": class_id_to_label,
        }
    })


# In[ ]:



run = wandb.init(project='kaggle-covid', 
                 config={'_wandb_kernel': 'tinghui'},
                 job_type='visualize_bbox')

wandb_bbox_list = []

for i in tqdm(range(lens)):
    row = test_meta_df.loc[i]
    id_name = row.image_id
    image = cv2.imread(os.path.join(TEST_PATH, f'{id_name}.jpg'))
    pred_bboxes = []
    pred_con = []
    pred_class = []
    if f'{id_name}.txt' in prediction_files:
        # opacity label
        pred_con, pred_bboxes, pred_class = get_pred(f'{PRED_PATH}/{id_name}.txt')
        
        pred_bboxes = scale_wandb_format(pred_bboxes)
        
        wandb_bbox_list.append(wandb_bbox(image, 
                                      pred_bboxes, 
                                      pred_class))
        
        
wandb.log({"radiograph": wandb_bbox_list})

run.finish()

run

