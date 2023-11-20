import pandas as pd
import os
from dotenv import load_dotenv
import numpy as np
from keras import models,layers
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
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

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    model = models.Sequential()
    model.add(layers.Dense(128, activation='relu', input_shape=(11,)))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(32, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    y_pred = model.predict(x_test)
    history = model.fit(x_train,y_train,epochs=5,batch_size=32,validation_data=(x_test,y_test))
    test_loss, test_accuracy = model.evaluate(x_test, y_test)
    conf_matrix = confusion_matrix(y_test, y_pred)
    print(f'Test Accuracy: {test_accuracy * 100:.2f}%')
    print("Confusion Matrix:")
    print(conf_matrix)
    return "Done"

print(nutrifact())