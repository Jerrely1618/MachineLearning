import pandas as pd
import os
from dotenv import load_dotenv
import numpy as np
from keras import models,layers
from sklearn.preprocessing import StandardScaler

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

def nutrifact():
    y = []
    scaler = StandardScaler()
    x = scaler.fit_transform(FOOD_SET[['calories', 'total_fat', 'saturated_fat', 'cholesterol', 'sodium', 'carbohydrate', 'fiber', 'sugars', 'calcium', 'iron', 'potassium']])

    
    for current_facts in x:
        product_values = np.zeros(11,)

        for i,(daily,fact) in enumerate(zip(DAILY_VALUES.values(),current_facts)):
            percentage = (fact / daily)
            product_values[i]=(round(percentage,5))
        
        product_unhealthy = np.sum(product_values > 0.2)

        y.append(1 if product_unhealthy == DETERMINANT else 0)

    y = np.array(y)

    model = models.Sequential()
    model.add(layers.Dense(128, activation='relu', input_shape=(11,)))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(32, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

nutrifact()