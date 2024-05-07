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
df1 = pd.read_csv("DATA_FULL_Tx1_WALLS_EMPTY.csv",header=None, names=series1)
df2 = pd.read_csv("DATA_FULL_Tx2_WALLS_EMPTY.csv",header=None, names=series1)
df3 = pd.read_csv("DATA_FULL_Tx3_WALLS_EMPTY.csv",header=None, names=series1)
df = pd.concat([df1, df2,df3])
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



# # #добавление номеров комнат
# def even_odd(x,y):
#     if (x<=5 and y<=5):
#         return 1
#     if (x<=5 and y>=7 and y<=11):
#         return 2
#     if (x<=5 and y>=13 and y<=17):
#         return 3
#     if (x<=5 and y>=19 and y<=24):
#         return 4
#     if (x>=1 and y>=25 and y<=31):
#         return 5
#     if (x>=11 and x<=15 and y>=19 and y<=24):
#         return 6
#     if (x>=11 and x<=15 and y>=13 and y<=17):
#         return 7
#     if (x>=11 and x<=15 and y>=7 and y<=11):
#         return 8
#     if (x>=11 and x<=15 and y>=1 and y<=5):
#         return 9
#     if (x>=7 and x<=9 and y<=23):
#         return 10

# x = df.loc[:,'x']
# y = df.loc[:,'y']
# N = list(map(even_odd,x,y))
# df.loc[:, "N"] = N

X_MSE_AV=[]
X_MAE_AV=[]
Y_MSE_AV=[]
Y_MAE_AV=[]
x_test=[]
y_test=[]
x_predict=[]
y_predict=[]
x_error=[]
y_error=[]
dir=[]

for i in range(0,373):
    df.reset_index(drop=True, inplace=True)
    test_data = df.iloc[[i]]
    train_data = df.drop(labels = [i])
    train_data.to_csv("train_data.csv",encoding = 'cp1251')
    
    # test_data = df.sample(n=1, replace=True)
    # train_data = df.drop(test_data.index)
    # train_data.to_csv("train_data.csv",encoding = 'cp1251')
    
    #отделить признаки от ярлыков
    train_labels = train_data.loc[:,('x','y')]
    test_labels = test_data.loc[:,('x','y')]
    
    #создание обучающих и тестовых наборов
    # test_data = df.sample(n=1, replace=True)
    # train_data = df.drop(test_data.index)
    # train_data.to_csv("train_data.csv",encoding = 'cp1251')
    
    
    
    #отделить признаки от ярлыков
    # train_labels = train_data.loc[:,('x','y')]
    # test_labels = test_data.loc[:,('x','y')]
    # test_labels.to_csv("test_labels.csv",encoding = 'cp1251')
    
    
    #мощность и АОD
    train_data = train_data.loc[:,("мощность 1-го луча", "мощность 2-го луча", "мощность 3-го луча", "мощность 4-го луча", "мощность 5-го луча", "мощность 6-го луча",  "мощность 7-го луча", "мощность 8-го луча", "мощность 9-го луча", "мощность 10-го луча", "мощность 11-го луча", "мощность 12-го луча", "мощность 13-го луча", "мощность 14-го луча", "мощность 15-го луча")]
    test_data = test_data.loc[:,("мощность 1-го луча", "мощность 2-го луча",  "мощность 3-го луча",  "мощность 4-го луча", "мощность 5-го луча","мощность 6-го луча",  "мощность 7-го луча", "мощность 8-го луча", "мощность 9-го луча", "мощность 10-го луча", "мощность 11-го луча", "мощность 12-го луча", "мощность 13-го луча", "мощность 14-го луча", "мощность 15-го луча")]
    
    
    train_stats = train_data.describe()
    train_stats = train_stats.transpose()
    
    def norm(x):
      return (x - train_stats['min']) /  (train_stats['max'] - train_stats['min'])
    
    inputs = keras.Input(shape=(len(train_data.keys())), name="digits")
    x = layers.Dense(75, activation="relu", name="dense_1")(inputs)
    x = layers.Dense(75, activation="relu", name="dense_2")(x)
    outputs = layers.Dense(2,  name="predictions")(x)
    model = keras.Model(inputs=inputs, outputs=outputs)
    model.compile(
    optimizer=keras.optimizers.RMSprop(),
    loss=['mse','mse'],
    metrics=['mae', 'mse'],)
    # посмотреть структуру нейронной сети
    # model.summary()
    EPOCHS = 400
    history = model.fit(
      norm(train_data),
      train_labels,
      epochs=EPOCHS,
      validation_split = 0.1,
      # shuffle = True,
      # validation_data=(test_data, test_labels),
      verbose=0)
    # plt.figure()
    # plt.plot(history.history['mae'])
    # plt.plot(history.history['val_mae'])
    # plt.title('Точность модели')
    # plt.ylabel('Средняя абсолютная ошибка')
    # plt.xlabel('Количество эпох')
    # plt.legend(['Тренировочный набор', 'Валидационный набор'], loc='upper left')
    # plt.grid(True)
    # plt.show()
    # plt.figure()
    # plt.plot(history.history['mse'])
    # plt.plot(history.history['val_mse'])
    # plt.title('Точность модели')
    # plt.ylabel('Средняя квадратичная ошибка')
    # plt.xlabel('Количество эпох')
    # plt.legend(['Тренировочный набор', 'Валидациооный набор'], loc='upper left')
    # plt.grid(True)
    # plt.show()
    
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
    # print('x: %.3f, y: %.3f' % (y_test_for_x, y_test_for_y))

    X_MAE_AV.append(x_mae)
    X_MSE_AV.append(x_mse)
    Y_MAE_AV.append(y_mae)
    Y_MSE_AV.append(y_mse)
    x_test.extend(y_test_for_x)
    y_test.extend(y_test_for_y)
    x_predict.extend(X_predict_for_x)
    y_predict.extend(X_predict_for_y)
    y_error.extend(y_err)
        
    d = (x_err**2+y_err**2)**1/2
    dir.extend(d)

    print(i)
    Results = [X_MSE_AV,X_MAE_AV,Y_MSE_AV,Y_MAE_AV,x_test,y_test,dir]
    Results = pd.DataFrame(Results)
    Results = Results.transpose()
    Results = round(Results,3)
    writer = pd.ExcelWriter('Results.xlsx')
    Results.to_excel(writer) 
    writer.save()


AV_X_MSE = round(sum(X_MSE_AV) / 374,3)
AV_X_MAE = round(sum(X_MAE_AV) / 374,3)
AV_Y_MSE = round(sum(Y_MSE_AV) / 374,3)
AV_Y_MAE = round(sum(Y_MAE_AV) / 374,3)
print(AV_X_MSE)
print(AV_X_MAE)
print(AV_Y_MSE)
print(AV_Y_MAE)

Results.columns=["X_MSE_AV","X_MAE_AV","Y_MSE_AV","Y_MAE_AV","x_test","y_test","dir"]
Results.to_csv("Results_nearest_neighbors.csv",encoding = 'cp1251')
df = Results[["x_test","y_test","dir"]]
df=df.pivot('x_test', 'y_test', 'dir')
ax = sns.heatmap(df,square=True,cmap = sns.cm.rocket_r)
plt.title('Геометрическое расстояние м/у предсказанной и измеренной точкой', fontsize=8)
plt.title('d')
plt.ylabel('x')
plt.xlabel('y')
plt.gca().invert_yaxis()
plt.show()