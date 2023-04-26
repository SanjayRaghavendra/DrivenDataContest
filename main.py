import os 
import numpy as np
import pandas as pd

import lightgbm as lgb

import keras 
from keras.layers import *
from keras.optimizers import *
from keras.models import Model
import argparse
from sklearn.metrics import f1_score 
from sklearn.model_selection import KFold


def ensemble_models(models, x):
    pred = []
    
    for model in models:
        y_pred = model.predict(x)
        pred.append(y_pred)
        
    temp = pred[0]
    for ypred in pred[1:]:
        temp += ypred
        
    y_pred = get_argmax(temp)
    
    return y_pred

def get_argmax(predicted_scores):
    res = []
    for _, score in enumerate(predicted_scores):
        idx = np.array(score).argmax(axis=0)
        temp = list(np.zeros((len(score))))
        temp[idx]=1
        res.append(temp)
        
    return np.array(res)


def make_parser():
    parser = argparse.ArgumentParser("Driven Data Competition") 
    parser.add_argument("--mode", default="eval")
    parser.add_argument("--data_path", default="./data")
    args = parser.parse_args()
    return args

def main():
    SEED = 1881
    args = make_parser()
    if args.mode == "eval":
        test_data = pd.read_csv(args.data_path)
        test_data = test_data.drop('Unnamed: 0', axis=1)
        models = []
        df = test_data.drop(["building_id"], axis=1)
        x = np.array(df)
        for i in range(5):
            model = lgb.Booster(model_file=f'models/model{i}.txt')
            y_pred = model.predict(x)
            models.append(model)
        
        y_pred = ensemble_models(models, x)
        y_pred = y_pred.argmax(axis=1)+1
        sub_csv = pd.read_csv("submissions/"+"submission_format.csv")
        sub_csv["damage_grade"] = y_pred
        sub_csv.to_csv("submissions/submission5.csv", index=False)


    elif args.mode == "train":
        train_data = pd.read_csv('/Users/sanjay/Downloads/train_values_4_19.csv')
        train_y = pd.read_csv('/Users/sanjay/Downloads/CS567-project/data/train_labels.csv')
        train_data = train_data.drop('Unnamed: 0', axis=1)

        y = np.array(train_y["damage_grade"])-1

        df = train_data.drop(["building_id"], axis=1)
        x = np.array(df)

        k_fold = KFold(n_splits=5, shuffle=True, random_state=SEED)
        for idx, (train_index, test_index) in enumerate(k_fold.split(x)):
            lgb_params = {
                "objective" : "multiclass",
                "num_class":3,
                "feature_fraction" : 0.5,
                "metric" : "multi_error",
                "boosting": 'gbdt',
                "seed": SEED,
                "max_depth" : -1,
                "min_sum_hessian_in_leaf" : 0.1,
                "max_bin":8192,
                "verbosity" : 1,
                "num_threads":6,
                "num_leaves" : 30,
                "learning_rate" : 0.1 
                
            }

            x_train, x_val, y_train, y_val= x[train_index], x[test_index], y[train_index], y[test_index]

            train_data = lgb.Dataset(x_train, label=y_train)
            val_data   = lgb.Dataset(x_val, label=y_val)

            lgb_model = lgb.train(lgb_params,
                                train_data,
                                20000,
                                valid_sets = [val_data],
                                verbose_eval = 1000,
                                early_stopping_rounds=3000)

            y_pred = lgb_model.predict(x_val)
            print("F1-MICRO SCORE: ", f1_score(np.array(pd.get_dummies(y_val)), get_argmax(y_pred), average='micro'))
            lgb_model.save_model(f'models/model{idx}.txt')

if __name__ == "__main__":
    main()