import pandas as pd
import os
from dotenv import load_dotenv
import numpy as np
from keras import models,layers
import time

load_dotenv()
FILE = os.environ.get('dataset_location')
FOOD_SET = pd.read_csv(FILE).fillna(0.0)
DETERMINANT = 2
DAILY_VALUES = {'calories': 2000.0,
                'total_fat': 65.0,
                'saturated_fat': 20.0,
                'cholesterol':300.0,
                'sodium':2300.0,
                'carbohydrate':225.0,
                'fiber':30.0,
                'sugars':50.0,
                'calcium':20.0,
                'iron': 18.0,
                'potassium':4700.0}
DIV = round(0.8 * len(FOOD_SET))

def timing_val(func):
    def wrapper(*arg, **kw):
        t1 = time.time()
        res = func(*arg, **kw)
        t2 = time.time()
        return (t2 - t1), res, func.__name__
    return wrapper

@timing_val
def nutrifact():
    x, y = [],[]
    data = FOOD_SET[['calories','total_fat','saturated_fat','cholesterol','sodium','carbohydrate','fiber','sugars','calcium','iron','potassium']]

    for current_facts in data.values:
        product_values = np.zeros(11,)

        for i,(daily,fact) in enumerate(zip(DAILY_VALUES.values(),current_facts)):
            percentage = (fact / daily)
            product_values[i]=(round(percentage,5))
        
        product_unhealthy = np.sum(product_values > 0.2)

        y.append(1 if product_unhealthy == DETERMINANT else 0)

        x.append(product_values)

    x_train,y_train = np.array(x[:DIV]),np.array(y[:DIV])
    x_test,y_test = np.array(x[DIV:]),np.array(y[DIV:])

    model = models.Sequential()
    model.add(layers.Dense(128, activation='relu', input_shape=(11,)))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(32, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    history = model.fit(x_train,y_train,epochs=5,batch_size=32,validation_data=(x_test,y_test))
    return "Done"

print(nutrifact())