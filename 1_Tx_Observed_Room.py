import os
import pathlib
import numpy as np
import matplotlib.pyplot as plt

import seaborn as sns
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from keras.layers import Dense
from keras import losses
from matplotlib import pyplot
from keras.models import Sequential
from sklearn import metrics
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
import pandas as pd
#открываем рабочий каталог ('cwd')
cwd=os.getcwd()
cwd

#изменение директории
os.chdir("C:\\Users\\Азалия\\Desktop\\Диплом\\Начальные данные\\Моя версия данных")


#информация об именах файлов
os.listdir('.')
#создание серий названий параметров
series1 = pd.Series(["фаза 1-го луча", "мощность 1-го луча", "время 1-го луча", "AоD phi 1-го луча", "AоD theta 1-го луча", "AоA phi 1-го луча", "AоА theta 1-го луча", "фаза 2-го луча", "мощность 2-го луча", "время 2-го луча", "AоD phi 2-го луча", "AоD theta 2-го луча", "AоA phi 2-го луча", "AоА theta 2-го луча", "фаза 3-го луча", "мощность 3-го луча", "время 3-го луча", "AоD phi 3-го луча", "AоD theta 3-го луча", "AоA phi 3-го луча", "AоА theta 3-го луча", "фаза 4-го луча", "мощность 4-го луча", "время 4-го луча", "AоD phi 4-го луча", "AоD theta 4-го луча", "AоA phi 4-го луча", "AоА theta 4-го луча", "фаза 5-го луча", "мощность 5-го луча", "время 5-го луча", "AоD phi 5-го луча", "AоD theta 5-го луча", "AоA phi 5-го луча", "AоА theta 5-го луча", "фаза 6-го луча", "мощность 6-го луча", "время 6-го луча", "AоD phi 6-го луча", "AоD theta 6-го луча", "AоA phi 6-го луча", "AоА theta 6-го луча", "фаза 7-го луча", "мощность 7-го луча", "время 7-го луча", "AоD phi 7-го луча", "AоD theta 7-го луча", "AоA phi 7-го луча", "AоА theta 7-го луча", "фаза 8-го луча", "мощность 8-го луча", "время 8-го луча", "AоD phi 8-го луча", "AоD theta 8-го луча", "AоA phi 8-го луча", "AоА theta 8-го луча", "фаза 9-го луча", "мощность 9-го луча", "время 9-го луча", "AоD phi 9-го луча", "AоD theta 9-го луча", "AоA phi 9-го луча", "AоА theta 9-го луча", "фаза 10-го луча", "мощность 10-го луча", "время 10-го луча", "AоD phi 10-го луча", "AоD theta 10-го луча", "AоA phi 10-го луча", "AоА theta 10-го луча", "фаза 11-го луча", "мощность 11-го луча", "время 11-го луча", "AоD phi 11-го луча", "AоD theta 11-го луча", "AоA phi 11-го луча", "AоА theta 11-го луча", "фаза 12-го луча", "мощность 12-го луча", "время 12-го луча", "AоD phi 12-го луча", "AоD theta 12-го луча", "AоA phi 12-го луча", "AоА theta 12-го луча", "фаза 13-го луча", "мощность 13-го луча", "время 13-го луча", "AоD phi 13-го луча", "AоD theta 13-го луча", "AоA phi 13-го луча", "AоА theta 13-го луча", "фаза 14-го луча", "мощность 14-го луча", "время 14-го луча", "AоD phi 14-го луча", "AоD theta 14-го луча", "AоA phi 14-го луча", "AоА theta 14-го луча", "фаза 15-го луча", "мощность 15-го луча", "время 15-го луча", "AоD phi 15-го луча", "AоD theta 15-го луча", "AоA phi 15-го луча", "AоА theta 15-го луча",'x','y','z'])

#создание структурированной таблицы данных                     
df = pd.read_csv("DATA_FULL_Tx3_WALLS_OBTACLES.csv",header=None, names=series1)
df = round(df,3)
#удаление точек в стенах
df = df.drop(df[(df.x==0)].index)
df = df.drop(df[(df.x==16)].index)
df = df.drop(df[(df.y==0)].index)
df = df.drop(df[(df.y==24)].index)
df = df.drop(df[(df.y==32)].index)
df = df.drop(df[(df.x==6)&(df.y<=24)].index)
df = df.drop(df[(df.x==10)&(df.y<=24)].index)
df = df.drop(df[((df.x<=6)|((df.x>=10)&(df.x<=16)))&((df.y==6)|(df.y==12)|(df.y==18))].index)



# #добавление номеров комнат
def even_odd(x,y):
    if (x<=5 and y<=5):
        return 1
    if (x<=5 and y>=7 and y<=11):
        return 2
    if (x<=5 and y>=13 and y<=17):
        return 3
    if (x<=5 and y>=19 and y<=24):
        return 4
    if (x>=1 and y>=25 and y<=31):
        return 5
    if (x>=11 and x<=15 and y>=19 and y<=24):
        return 6
    if (x>=11 and x<=15 and y>=13 and y<=17):
        return 7
    if (x>=11 and x<=15 and y>=7 and y<=11):
        return 8
    if (x>=11 and x<=15 and y>=1 and y<=5):
        return 9
    if (x>=7 and x<=9 and y<=23):
        return 10

x = df.loc[:,'x']
y = df.loc[:,'y']
N = list(map(even_odd,x,y))
df.loc[:, "N"] = N


#создание обучающих и тестовых наборов
test_data = df.sample(n=1, replace=True)
train_data = df.drop(test_data.index)
train_data.to_csv("train_data.csv",encoding = 'cp1251')



#отделить признаки от ярлыков
train_labels = train_data.loc[:,('x','y')]
test_labels = test_data.loc[:,('x','y')]
# test_labels.to_csv("test_labels.csv",encoding = 'cp1251')


#мощность и АОD
train_data = train_data.loc[:,("мощность 1-го луча", "мощность 2-го луча", "мощность 3-го луча", "мощность 4-го луча", "мощность 5-го луча", "мощность 6-го луча",  "мощность 7-го луча", "мощность 8-го луча", "мощность 9-го луча", "мощность 10-го луча", "мощность 11-го луча", "мощность 12-го луча", "мощность 13-го луча", "мощность 14-го луча", "мощность 15-го луча","AоD phi 1-го луча", "AоD theta 1-го луча","AоD phi 2-го луча", "AоD theta 2-го луча","AоD phi 3-го луча", "AоD theta 3-го луча", "AоD phi 4-го луча", "AоD theta 4-го луча", "AоD phi 5-го луча", "AоD theta 5-го луча", "AоD phi 6-го луча", "AоD theta 6-го луча", "AоD phi 7-го луча", "AоD theta 7-го луча", "AоD phi 8-го луча", "AоD theta 8-го луча",  "AоD phi 9-го луча", "AоD theta 9-го луча", "AоD phi 10-го луча", "AоD theta 10-го луча", "AоD phi 11-го луча", "AоD theta 11-го луча", "AоD phi 12-го луча", "AоD theta 12-го луча", "AоD phi 13-го луча", "AоD theta 13-го луча", "AоA phi 13-го луча", "AоD phi 14-го луча", "AоD theta 14-го луча", "AоD phi 15-го луча", "AоD theta 15-го луча","N")]
test_data = test_data.loc[:,("мощность 1-го луча", "мощность 2-го луча",  "мощность 3-го луча",  "мощность 4-го луча", "мощность 5-го луча","мощность 6-го луча",  "мощность 7-го луча", "мощность 8-го луча", "мощность 9-го луча", "мощность 10-го луча", "мощность 11-го луча", "мощность 12-го луча", "мощность 13-го луча", "мощность 14-го луча", "мощность 15-го луча","AоD phi 1-го луча", "AоD theta 1-го луча","AоD phi 2-го луча", "AоD theta 2-го луча","AоD phi 3-го луча", "AоD theta 3-го луча", "AоD phi 4-го луча", "AоD theta 4-го луча", "AоD phi 5-го луча", "AоD theta 5-го луча", "AоD phi 6-го луча", "AоD theta 6-го луча", "AоD phi 7-го луча", "AоD theta 7-го луча", "AоD phi 8-го луча", "AоD theta 8-го луча",  "AоD phi 9-го луча", "AоD theta 9-го луча", "AоD phi 10-го луча", "AоD theta 10-го луча", "AоD phi 11-го луча", "AоD theta 11-го луча", "AоD phi 12-го луча", "AоD theta 12-го луча", "AоD phi 13-го луча", "AоD theta 13-го луча", "AоA phi 13-го луча", "AоD phi 14-го луча", "AоD theta 14-го луча", "AоD phi 15-го луча", "AоD theta 15-го луча","N")]


train_stats = train_data.describe()
train_stats = train_stats.transpose()

#нормализация входных данных
def norm(x):
  return (x - train_stats['min']) /  (train_stats['max'] - train_stats['min'])
# norm_train_data = norm(train_data)
# norm_train_data = norm_train_data.describe()
# norm_train_data = norm_train_data.transpose()
# print('Максимальные значения нормированных параметров:')
# print(norm_train_data['max'])
# print('Минимальные значения нормированных параметров:')
# print(norm_train_data['min'])

# создание модели нейронной сети
inputs = keras.Input(shape=(len(train_data.keys())), name="digits")
x = layers.Dense(75, activation="relu", name="dense_1")(inputs)
x = layers.Dense(75, activation="relu", name="dense_2")(x)
x = layers.Dense(75, activation="relu", name="dense_3")(x)

outputs = layers.Dense(2,  name="predictions")(x)
model = keras.Model(inputs=inputs, outputs=outputs)
model.compile(
optimizer=keras.optimizers.RMSprop(),
loss=['mse','mse'],
metrics=['mae', 'mse'])

# model.summary()

EPOCHS = 500

history = model.fit(
  norm(train_data),
  train_labels,
  epochs=EPOCHS,
  validation_split = 0.2,
  # shuffle = True,
  # validation_data=(test_data, test_labels),
  verbose=0)



plt.figure()
plt.plot(history.history['mae'])
plt.plot(history.history['val_mae'])

plt.title('Точность модели')
plt.ylabel('Средняя абсолютная ошибка')
plt.xlabel('Количество эпох')
plt.legend(['Тренировочный набор', 'Валидационный набор'], loc='upper left')
plt.grid(True)
plt.show()

plt.figure()
plt.plot(history.history['mse'])
plt.plot(history.history['val_mse'])

plt.title('Точность модели')
plt.ylabel('Средняя квадратичная ошибка')
plt.xlabel('Количество эпох')
plt.legend(['Тренировочный набор', 'Валидационый набор'], loc='upper left')
plt.grid(True)
plt.show()



y_test_for_x = np.array(test_labels.loc[:,'x'])
X_predict_for_x = np.array(model.predict(norm(test_data))[:,0])
x_err = abs(y_test_for_x-X_predict_for_x)

y_test_for_y = np.array(test_labels.loc[:,'y'])
X_predict_for_y = np.array(model.predict(norm(test_data))[:,1])
y_err = abs(y_test_for_y-X_predict_for_y)
# Среднеквадратическая ошибка
x_mse = metrics.mean_squared_error(X_predict_for_x, y_test_for_x)
# Абсолютная ошибка
x_mae = metrics.mean_absolute_error(X_predict_for_x, y_test_for_x)
print('x_mse: %.3f, x_mae: %.3f' % (x_mse, x_mae))
# Среднеквадратическая ошибка
y_mse = metrics.mean_squared_error(X_predict_for_y, y_test_for_y)
# Абсолютная ошибка
y_mae = metrics.mean_absolute_error(X_predict_for_y, y_test_for_y)
print('y_mse: %.3f, y_mae: %.3f' % (y_mse, y_mae))



plt.figure()
plt.scatter(y_test_for_x,X_predict_for_x)
plt.scatter(y_test_for_y,X_predict_for_y)
plt.ylabel('x_err or y_err')
plt.xlabel('x or y')
plt.legend(['x_err(x)', 'y_err(y)'], loc='upper left')
plt.show()


d = (x_err**2+y_err**2)**1/2

y_test_for_x = pd.Series(y_test_for_x)
X_predict_for_x = pd.Series(X_predict_for_x)
y_test_for_y = pd.Series(y_test_for_y)
X_predict_for_y = pd.Series(X_predict_for_y)
d = pd.Series(d)
massiv = pd.DataFrame({'x': y_test_for_x,'y': y_test_for_y,'x_err': X_predict_for_x,'y_err': X_predict_for_y,'d': d})


D = d.to_numpy()
X = massiv.loc[:,'x']
X = X.to_numpy()
Y = massiv.loc[:,'y']
Y = Y.to_numpy()

array = np.zeros((17,33))
array[X,Y] = d
array = np.transpose(array)

# print(array)

import seaborn as sns
import matplotlib.pyplot as plt

sns.heatmap(array,square=True,cmap = sns.cm.rocket_r)
plt.title('Геометрическое расстояние м/у предсказанной и измеренной точкой', fontsize=8)
plt.title('d')
plt.ylabel('y')
plt.xlabel('x')
plt.gca().invert_yaxis()
plt.show()





