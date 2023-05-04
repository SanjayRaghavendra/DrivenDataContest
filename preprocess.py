#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  3 11:15:25 2023

@author: seanfoley
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from scipy.stats.mstats import winsorize

import os 


import keras 
from keras.layers import *
from keras.optimizers import *
from keras.models import Model



train_labels = pd.read_csv('/data/train_labels.csv')
train_values = pd.read_csv('/data/train_values.csv')
test_values = pd.read_csv('/data/test_values.csv')


#separate data into numerical, categorical, and binary features
#geo 1 = city; geo 2 = district; geo 3 = street
geo = ['geo_level_1_id', 'geo_level_2_id', 'geo_level_3_id']

geo_2 = ['geo_level_2_id', 'geo_level_3_id']

num=['count_floors_pre_eq','area_percentage', 'age','height_percentage','count_families']
cat=['land_surface_condition', 'foundation_type', 'roof_type',
       'ground_floor_type', 'other_floor_type', 'position','legal_ownership_status',
       'plan_configuration','geo_level_1_id', 'geo_level_2_id', 'geo_level_3_id','city_freq','district_freq','street_freq','city_sd','district_sd','street_sd'] 
cat2=['land_surface_condition', 'foundation_type', 'roof_type',
       'ground_floor_type', 'other_floor_type', 'position','legal_ownership_status',
       'plan_configuration']
binary=['has_superstructure_adobe_mud',
       'has_superstructure_mud_mortar_stone', 'has_superstructure_stone_flag',
       'has_superstructure_cement_mortar_stone',
       'has_superstructure_mud_mortar_brick',
       'has_superstructure_cement_mortar_brick', 'has_superstructure_timber',
       'has_superstructure_bamboo', 'has_superstructure_rc_non_engineered',
       'has_superstructure_rc_engineered', 'has_superstructure_other',
         'has_secondary_use_hotel',
       'has_secondary_use_rental', 'has_secondary_use_institution',
       'has_secondary_use_school', 'has_secondary_use_industry',
       'has_secondary_use_health_post', 'has_secondary_use_gov_office',
       'has_secondary_use_use_police', 'has_secondary_use_other']


#one hot encode categorical variables
train_values = pd.get_dummies(train_values,columns=cat2)
test_values = pd.get_dummies(test_values,columns=cat2)

#scaler for continuous data
#train_values[num] = train_values.loc[:, num]\
    #.apply(lambda x: np.log(x+1))

scaler = StandardScaler()

for col in num:
  train_values[col] = scaler.fit_transform(train_values[col].values.reshape(-1,1))
  test_values[col] = scaler.fit_transform(test_values[col].values.reshape(-1,1))
  

#account for outliers in numerical features
for col in num:
  train_values[col] = winsorize(train_values[col])
  test_values[col] = winsorize(test_values[col])
  
##preprocessing for geo features##

#geo feats
geo1 = np.array(pd.get_dummies(pd.concat([train_values["geo_level_1_id"], test_values["geo_level_1_id"]])))
geo2 = np.array(pd.get_dummies(pd.concat([train_values["geo_level_2_id"], test_values["geo_level_2_id"]])))
geo3 = np.array(pd.get_dummies(pd.concat([train_values["geo_level_3_id"], test_values["geo_level_3_id"]])))

def geo_mod():
    inp = Input((geo3.shape[1],))
    i1 = Dense(16, name="intermediate")(inp)
    x2 = Dense(geo2.shape[1], activation='sigmoid')(i1)
    x1 = Dense(geo1.shape[1], activation='sigmoid')(i1)

    model = Model(inp, [x2,x1])
    model.compile(loss="binary_crossentropy", optimizer="adam")
    return model

model = geo_mod()
model.fit(geo3, [geo2, geo1], batch_size=128, epochs=10, verbose=2)
model.save("geo_feats.h5")

model = geo_mod()
model.load_weights('geo_feats.h5')

# "Extract Intermediate Layer" Function
from keras import backend as K

get_int_layer_output = K.function([model.layers[0].input],
                                  [model.layers[1].output])

out = []
for dat in geo3[:260601]:
    dat = np.reshape(dat, (1, dat.shape[0]))
    layer_output = get_int_layer_output([[dat]])[0]
    out.append(layer_output)

out = np.array(out)
out = np.squeeze(out)

train_data = train_values.copy()
train_data = train_data.drop(['geo_level_1_id', 'geo_level_2_id', 'geo_level_3_id'], axis=1)
train_data = train_data.assign(geo_feat1=out[:,0],
                               geo_feat2=out[:,1],
                               geo_feat3=out[:,2],  
                               geo_feat4=out[:,3],
                               geo_feat5=out[:,4],    
                               geo_feat6=out[:,5],
                               geo_feat7=out[:,6],
                               geo_feat8=out[:,7],
                               geo_feat9=out[:,8],
                               geo_feat10=out[:,9],
                               geo_feat11=out[:,10],
                               geo_feat12=out[:,11],
                               geo_feat13=out[:,12],
                               geo_feat14=out[:,13],
                               geo_feat15=out[:,14],           
                               geo_feat16=out[:,15])


# Extract GEO-Embeds for all test data points.
# Then assign with test_data

out = []
for dat in geo3[260601:]:
    dat = np.reshape(dat, (1, dat.shape[0]))
    layer_output = get_int_layer_output([[dat]])[0]
    out.append(layer_output)

out = np.array(out)
out = np.squeeze(out)

test_data = test_values.copy()
test_data = test_data.drop(['geo_level_1_id', 'geo_level_2_id', 'geo_level_3_id'], axis=1)
test_data = test_data.assign(geo_feat1=out[:,0],
                             geo_feat2=out[:,1],
                             geo_feat3=out[:,2],  
                             geo_feat4=out[:,3],
                             geo_feat5=out[:,4],    
                             geo_feat6=out[:,5],
                             geo_feat7=out[:,6],
                             geo_feat8=out[:,7],
                             geo_feat9=out[:,8],
                             geo_feat10=out[:,9],
                             geo_feat11=out[:,10],
                             geo_feat12=out[:,11],
                             geo_feat13=out[:,12],
                             geo_feat14=out[:,13],
                             geo_feat15=out[:,14],           
                             geo_feat16=out[:,15])

train_data.to_csv('train_values_final.csv')
test_data.to_csv('test_values_final.csv')
