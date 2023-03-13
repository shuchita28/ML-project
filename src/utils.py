import os
import sys

import pandas as pd
import numpy as np
import dill
from time import time

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from src.exception import CustomException

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)
    except Exception as e:
        raise CustomException(e, sys)
    
def evaluate_model(X_train, y_train, X_test, y_test, models):
    print('_' * 80)
    print("Training: ")
    print(models)
    t0 = time()
    models.fit(X_train, y_train)
    train_time = time() - t0
    print("train time: %0.3fs" % train_time)

    t0 = time()
    pred = models.predict(X_test)
    test_time = time() - t0
    print("test time:  %0.3fs" % test_time)

    score = accuracy_score(y_test, pred)
    print("accuracy:   %0.3f" % score)

    if hasattr(models, 'coef_'):
        print("dimensionality: %d" % models.coef_.shape[1])
        print("density: %f" % density(models.coef_))

    print("classification report:")
    print(classification_report(y_test, pred))

    print("confusion matrix:")
    print(confusion_matrix(y_test, pred))

    clf_descr = str(models).split('(')[0]
    return clf_descr, score, train_time, test_time
    