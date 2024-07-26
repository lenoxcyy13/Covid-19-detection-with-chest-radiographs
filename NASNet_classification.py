#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import tensorflow as tf
print(tf.__version__)
from tensorflow.keras import layers
from tensorflow.keras import models
import tensorflow_addons as tfa
from tensorflow.keras import mixed_precision


import tensorflow_probability as tfp
tfd = tfp.distributions

import os
import gc
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

from sklearn.model_selection import StratifiedKFold
from sklearn.utils.class_weight import compute_class_weight

# Imports for augmentations. 
from albumentations import (Compose, RandomResizedCrop, Cutout, Rotate, HorizontalFlip, 
                            VerticalFlip, RandomBrightnessContrast, ShiftScaleRotate, 
                            CenterCrop, Resize)


# In[ ]:


# W&B related imports
import wandb
print(wandb.__version__)
from wandb.keras import WandbCallback

wandb.login()


# In[ ]:


# Increase GPU memory as per the need.
gpus = tf.config.list_physical_devices('GPU')
if gpus:
  try:
    # Currently, memory growth needs to be the same across GPUs
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Memory growth must be set before GPUs have been initialized
    print(e)


# # Hyperparameters

# In[ ]:


AUTOTUNE = tf.data.AUTOTUNE

CONFIG = dict (
    seed = 42,
    num_labels = 4,
    num_folds = 5,
    img_width = 224, # If you change the resolution to 512 reduce batch size. 
    img_height = 224,
    batch_size = 32,
    epochs = 100,
    learning_rate = 1e-4,
    architecture = "CNN",
    competition = 'siim-covid',
    infra = "GCP",
)


# # Merge CSV

# In[ ]:


# Load csv file
df_image = pd.read_csv('../covid/train_image_level.csv')
df_label = pd.read_csv('../covid/train_study_level.csv')
df_meta = pd.read_csv('../covid/meta.csv')

# Modify values in the id column
df_image['id'] = df_image.apply(lambda row: row.id.split('_')[0], axis=1)
# Add absolute path
df_image['path'] = df_image.apply(lambda row: '../covid2/train/'+row.id+'.png', axis=1)
# Get image level labels
df_image['image_level'] = df_image.apply(lambda row: row.label.split(' ')[0], axis=1)

df_image.head(5)


# In[ ]:


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


# In[ ]:


# Original dimensions 
train_meta_df = df_meta.loc[df_meta.split == 'train']
test_meta_df = df_meta.loc[df_meta.split == 'test']
train_meta_df = train_meta_df.drop('split', axis=1)
train_meta_df.columns = ['id', 'dim0', 'dim1']

train_meta_df.head(2)


# In[ ]:


# Merge both the dataframes
df = df.merge(train_meta_df, on='id',how="left")
df.to_csv('total.csv', index=False)
df.head(2)


# In[ ]:


df_total = pd.read_csv('total.csv')
# clear the date that has be labeled non-negative but no bbox
for i in range(len(df_total)):
    #print(row)
    row = df_total.loc[df_total.index.values[i]]
    label = row.study_level
    box = row.boxes

    if label != 'Negative' and box is np.nan:
        df = df.drop(df_total.index.values[i])
        #df = df.drop(grp_df.index.values[i])
        #print(len(df_total))
    
print(len(df_total), len(df))
df = df.drop('boxes', axis=1).reset_index()

df.head()


# ## Train-validation split

# In[ ]:


# Create train and validation split.
train_df, valid_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df.image_level.values)

train_df.loc[:, 'split'] = 'train'
valid_df.loc[:, 'split'] = 'valid'

df = pd.concat([train_df, valid_df]).reset_index(drop=True)


# In[ ]:


# Compute Class weights to mitigate class imbalance to some extent. 

class_weights = compute_class_weight('balanced', 
                                    classes=np.unique(train_df['study_level'].values),
                                    y=train_df['study_level'].values)

class_weights_dict = {key: val for key, val in zip(np.unique(train_df['study_level'].values), class_weights)}
class_weights_dict                                                            


# ## Augmentation

# In[ ]:


@tf.function
def decode_image(image):
    # convert the compressed string to a 3D uint8 tensor
    image = tf.image.decode_png(image, channels=3)
    # Normalize image
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    return image

def load_image(df_dict):
    # Load image
    image = tf.io.read_file(df_dict['path'])
    image = decode_image(image)
    
    # Parse label
    label = df_dict['study_level']
    label = tf.one_hot(indices=label, depth=CONFIG['num_labels'])
    
    return image, label

# Mixup Augmentation policy
@tf.function
def mixup(a, b):
    alpha = [1.0]
    beta = [1.0]

    # unpack (image, label) pairs
    (image1, label1), (image2, label2) = a, b

    # define beta distribution
    dist = tfd.Beta(alpha, beta)
    # sample from this distribution
    l = dist.sample(1)[0][0]

    # mixup augmentation
    img = l*image1+(1-l)*image2
    lab = l*label1+(1-l)*label2

    return img, lab


# In[ ]:


CROP_SIZE = CONFIG['img_height']
'''
# Random Resized Crop
transforms = Compose([
            HorizontalFlip(p=0.6),
#             Rotate(limit=6, p=0.3),
            ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=5, p=0.6),
            RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, brightness_by_max=False, p=0.5),
#             Cutout(num_holes=2, max_h_size=int(0.4*CONFIG['img_height']), max_w_size=int(0.4*CONFIG['img_height']), fill_value=0, always_apply=False, p=1.0)
        ])
'''

# Imports for augmentations. 
import albumentations as A
transforms = A.Compose(
    [
            #A.HorizontalFlip(p=0.6),
#             Rotate(limit=6, p=0.3),
            #A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=5, p=0.6),
            
            #A.Cutout(num_holes=2, max_h_size=int(0.4*CONFIG['img_height']), max_w_size=int(0.4*CONFIG['img_height']), fill_value=0, always_apply=False, p=1.0),
            A.Flip(),
            A.CenterCrop(height =CONFIG['img_height'],width =CONFIG['img_height']),
            A.RandomRotate90(p=0.15),
            #A.RandomTranslation(height_factor=0.1, width_factor=0.1),
            
            #A.Sharpen(alpha = (0.2,0.5), lightness = (0.5, 1), always_apply = False, p =0.5),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, brightness_by_max=False, p=0.5),
            #A.RandomContrast(factor=0.1),
        ]
)
def aug_fn(image):
    data = {"image":image}
    aug_data = transforms(**data)
    aug_img = aug_data["image"]

    return aug_img.astype(np.float32) 

def augmentations(image, label):
    aug_img = tf.numpy_function(func=aug_fn, inp=[image], Tout=tf.float32)
    aug_img.set_shape((CROP_SIZE, CROP_SIZE, 3))

    return aug_img, label


# In[ ]:


AUTOTUNE = tf.data.AUTOTUNE

# Simple dataloader
def get_dataloaders(train_df, valid_df):
    trainloader = tf.data.Dataset.from_tensor_slices(dict(train_df))
    validloader = tf.data.Dataset.from_tensor_slices(dict(valid_df))

    trainloader = (
        trainloader
        .shuffle(1024)
        .map(load_image, num_parallel_calls=AUTOTUNE)
        .map(augmentations, num_parallel_calls=AUTOTUNE)
        .batch(CONFIG['batch_size'])
        .prefetch(AUTOTUNE)
    )

    validloader = (
        validloader
        .map(load_image, num_parallel_calls=AUTOTUNE)
        .batch(CONFIG['batch_size'])
        .prefetch(AUTOTUNE)
    )
    
    return trainloader, validloader

# Mixup
def get_mixup_dataloaders(train_df, valid_df):
    trainloader1 = tf.data.Dataset.from_tensor_slices(dict(train_df)).shuffle(1024).map(load_image, num_parallel_calls=AUTOTUNE)
    trainloader2 = tf.data.Dataset.from_tensor_slices(dict(train_df)).shuffle(1024).map(load_image, num_parallel_calls=AUTOTUNE)

    trainloader = tf.data.Dataset.zip((trainloader1, trainloader2))

    # Valid Loader
    validloader = tf.data.Dataset.from_tensor_slices(dict(valid_df))

    trainloader = (
        trainloader
        .shuffle(1024)
        .map(mixup, num_parallel_calls=AUTOTUNE)
        .map(augmentations, num_parallel_calls=AUTOTUNE)
        .batch(CONFIG['batch_size'])
        .prefetch(AUTOTUNE)
    )

    validloader = (
        validloader
        .map(load_image, num_parallel_calls=AUTOTUNE)
        .batch(CONFIG['batch_size'])
        .prefetch(AUTOTUNE)
    )
    
    return trainloader, validloader


# # get Model

# In[ ]:


def get_model():
    base_model = tf.keras.applications.nasnet.NASNetMobile(include_top=False, weights='imagenet')
    base_model.trainabe = True

    inputs = layers.Input((CONFIG['img_height'], CONFIG['img_width'], 3))
    x = base_model(inputs, training=True)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.5)(x)
    
    outputs = layers.Dense(CONFIG['num_labels'], kernel_regularizer=tf.keras.regularizers.l2(0.001))(x)
    outputs = layers.Activation('softmax', dtype='float32', name='predictions')(outputs)
    
    return models.Model(inputs, outputs)

tf.keras.backend.clear_session() 
model = get_model()
model.summary()


# In[ ]:


CONFIG['model_name'] = 'nasnet_2'


# # Train

# In[ ]:


# Early stopping regularization
'''
earlystopper = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss', patience=6, verbose=0, mode='min',
    restore_best_weights=True
)
'''

# Reduce learning rate when validation loss gets plateau 
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1,
                              patience=3, min_lr=CONFIG['learning_rate'])


# In[ ]:


# utility to run prediction on out-of-fold validation data. 
def get_predictions(model, validloader, valid_df):
    y_pred = []
    for image_batch, label_batch in tqdm(validloader):
        preds = model.predict(image_batch)
        y_pred.extend(preds)
        
    valid_df['preds'] = y_pred
    
    return valid_df 

# dataframe to collect oof predictions
oof_df = pd.DataFrame()


# In[ ]:


# Prepare dataloaders
trainloader, validloader = get_mixup_dataloaders(train_df, valid_df)
    
# Initialize model
tf.keras.backend.clear_session()
model = get_model()

# Compile model
optimizer = tf.keras.optimizers.Adam(learning_rate=CONFIG['learning_rate'])
model.compile(optimizer, 
              loss='categorical_crossentropy', 
               metrics=['acc', tf.keras.metrics.AUC(curve='ROC')])


# Update CONFIG dict with the name of the model.
print('Training configuration: ', CONFIG)

# Initialize W&B run
run = wandb.init(project='Study-level', 
        config=CONFIG,
        job_type='train'
                )
    
# Train
_ = model.fit(trainloader, 
            epochs=CONFIG['epochs'],
            validation_data=validloader,
            class_weight=class_weights_dict,
            callbacks=[WandbCallback(),
                    #earlystopper,
                    reduce_lr])
    
# Evaluate
loss, acc, auc = model.evaluate(validloader)
wandb.log({'Val Acc': acc, 'Val AUC-ROC': auc})
    
# Save model
model_name = CONFIG['model_name']
MODEL_PATH = f'models/study_level/{model_name}'
os.makedirs(MODEL_PATH, exist_ok=True)
count_models = len(os.listdir(MODEL_PATH))
    
model.save(f'{MODEL_PATH}/{model_name}_{count_models}.h5')

# Get Prediction on validation set
_oof_df = get_predictions(model, validloader, valid_df)
oof_df = pd.concat([oof_df, _oof_df])

# Close W&B run
run.finish()
    
del model, trainloader, validloader, _oof_df
_ = gc.collect()
    
os.makedirs('study_level_oof', exist_ok=True)
oof_df.to_csv('study_level_oof/oof_preds.csv', index=False)


# # CV Score

# In[ ]:


oof_df = pd.read_csv('study_level_oof/oof_preds.csv')
oof_df.head()


# In[ ]:


def get_argmax(row):
    return [float(val) for val in row.preds.strip('[]').split(' ') if val!='']

oof_df['preds'] = oof_df.apply(lambda row: get_argmax(row), axis=1)


# In[ ]:


metric = tf.keras.metrics.CategoricalAccuracy()
metric.update_state(tf.one_hot(oof_df.study_level.values, depth=4), tf.cast(np.array(list(map(np.array, oof_df.preds.values))), tf.float32))
print(f'CV Score: {metric.result().numpy()}')


# In[ ]:


metric = tf.keras.metrics.AUC(curve='ROC')
metric.update_state(tf.one_hot(oof_df.study_level.values, depth=4), tf.cast(np.array(list(map(np.array, oof_df.preds.values))), tf.float32))
print(f'CV Score: {metric.result().numpy()}')

