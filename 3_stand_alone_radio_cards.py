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
import random
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

df_Tx1 = pd.read_csv("DATA_FULL_Tx1_WALLS_OBTACLES.csv",header=None, names=series1)
df_Tx2 = pd.read_csv("DATA_FULL_Tx2_WALLS_OBTACLES.csv",header=None, names=series1)
df_Tx3 = pd.read_csv("DATA_FULL_Tx3_WALLS_OBTACLES.csv",header=None, names=series1)
df_Tx1 = round(df_Tx1,3)
df_Tx2 = round(df_Tx2,3)
df_Tx3 = round(df_Tx3,3)


df_Tx1 = df_Tx1.drop(df_Tx1[(df_Tx1.x==0)].index)
df_Tx1 = df_Tx1.drop(df_Tx1[(df_Tx1.x==16)].index)
df_Tx1 = df_Tx1.drop(df_Tx1[(df_Tx1.y==0)].index)
df_Tx1 = df_Tx1.drop(df_Tx1[(df_Tx1.y==24)].index)
df_Tx1 = df_Tx1.drop(df_Tx1[(df_Tx1.y==32)].index)
df_Tx1 = df_Tx1.drop(df_Tx1[(df_Tx1.x==6)&(df_Tx1.y<=24)].index)
df_Tx1 = df_Tx1.drop(df_Tx1[(df_Tx1.x==10)&(df_Tx1.y<=24)].index)
df_Tx1 = df_Tx1.drop(df_Tx1[((df_Tx1.x<=6)|((df_Tx1.x>=10)&(df_Tx1.x<=16)))&((df_Tx1.y==6)|(df_Tx1.y==12)|(df_Tx1.y==18))].index)

df_Tx2 = df_Tx2.drop(df_Tx2[(df_Tx2.x==0)].index)
df_Tx2 = df_Tx2.drop(df_Tx2[(df_Tx2.x==16)].index)
df_Tx2 = df_Tx2.drop(df_Tx2[(df_Tx2.y==0)].index)
df_Tx2 = df_Tx2.drop(df_Tx2[(df_Tx2.y==24)].index)
df_Tx2 = df_Tx2.drop(df_Tx2[(df_Tx2.y==32)].index)
df_Tx2 = df_Tx2.drop(df_Tx2[(df_Tx2.x==6)&(df_Tx2.y<=24)].index)
df_Tx2 = df_Tx2.drop(df_Tx2[(df_Tx2.x==10)&(df_Tx2.y<=24)].index)
df_Tx2 = df_Tx2.drop(df_Tx2[((df_Tx2.x<=6)|((df_Tx2.x>=10)&(df_Tx2.x<=16)))&((df_Tx2.y==6)|(df_Tx2.y==12)|(df_Tx2.y==18))].index)

df_Tx3 = df_Tx3.drop(df_Tx3[(df_Tx3.x==0)].index)
df_Tx3 = df_Tx3.drop(df_Tx3[(df_Tx3.x==16)].index)
df_Tx3 = df_Tx3.drop(df_Tx3[(df_Tx3.y==0)].index)
df_Tx3 = df_Tx3.drop(df_Tx3[(df_Tx3.y==24)].index)
df_Tx3 = df_Tx3.drop(df_Tx3[(df_Tx3.y==32)].index)
df_Tx3 = df_Tx3.drop(df_Tx3[(df_Tx3.x==6)&(df_Tx3.y<=24)].index)
df_Tx3 = df_Tx3.drop(df_Tx3[(df_Tx3.x==10)&(df_Tx3.y<=24)].index)
df_Tx3 = df_Tx3.drop(df_Tx3[((df_Tx3.x<=6)|((df_Tx3.x>=10)&(df_Tx3.x<=16)))&((df_Tx3.y==6)|(df_Tx3.y==12)|(df_Tx3.y==18))].index)



df_Tx1.reset_index(drop=True, inplace=True)
df1=df_Tx1

df_Tx2.reset_index(drop=True, inplace=True)
df2=df_Tx2

df_Tx3.reset_index(drop=True, inplace=True)
df3=df_Tx3

#  добавление номеров комнат
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


x = df_Tx1.loc[:,'x']
y = df_Tx1.loc[:,'y']
N1 = list(map(even_odd,x,y))
df_Tx1.loc[:, "N1"] = N1

x = df_Tx2.loc[:,'x']
y = df_Tx2.loc[:,'y']
N2 = list(map(even_odd,x,y))
df_Tx2.loc[:, "N2"] = N2

x = df_Tx3.loc[:,'x']
y = df_Tx3.loc[:,'y']
N3 = list(map(even_odd,x,y))
df_Tx3.loc[:, "N3"] = N3


X_MSE_AV_1=[]
X_MAE_AV_1=[]
Y_MSE_AV_1=[]
Y_MAE_AV_1=[]
x_test_1=[]
y_test_1=[]
x_predict_1=[]
y_predict_1=[]
x_error_1=[]
y_error_1=[]
dir_1=[]
av_dir_1=[]

X_MSE_AV_2=[]
X_MAE_AV_2=[]
Y_MSE_AV_2=[]
Y_MAE_AV_2=[]
x_test_2=[]
y_test_2=[]
x_predict_2=[]
y_predict_2=[]
x_error_2=[]
y_error_2=[]
dir_2=[]
av_dir_2=[]

X_MSE_AV_3=[]
X_MAE_AV_3=[]
Y_MSE_AV_3=[]
Y_MAE_AV_3=[]
x_test_3=[]
y_test_3=[]
x_predict_3=[]
y_predict_3=[]
x_error_3=[]
y_error_3=[]
dir_3=[]
av_dir_3=[]

k=373
for i in range(0,k):
        test_data = df1.iloc[[i]]
        train_data  = df_Tx1.query("index != @i")
        train_data.to_csv("train_data.csv",encoding = 'cp1251')
        
        #отделить признаки от ярлыков
        train_labels = train_data.loc[:,('x','y')]
        test_labels = test_data.loc[:,('x','y')]
        
        #создание обучающих и тестовых наборов
        # test_data = df.sample(n=1, replace=True)
        # train_data = df.drop(test_data.index)
        # train_data.to_csv("train_data.csv",encoding = 'cp1251')
        
        
        #мощность и АОD
        train_data = train_data.loc[:,("мощность 1-го луча", "мощность 2-го луча", "мощность 3-го луча", "мощность 4-го луча", "мощность 5-го луча", "мощность 6-го луча",  "мощность 7-го луча", "мощность 8-го луча", "мощность 9-го луча", "мощность 10-го луча", "мощность 11-го луча", "мощность 12-го луча", "мощность 13-го луча", "мощность 14-го луча", "мощность 15-го луча","AоD phi 1-го луча", "AоD theta 1-го луча","AоD phi 2-го луча", "AоD theta 2-го луча","AоD phi 3-го луча", "AоD theta 3-го луча", "AоD phi 4-го луча", "AоD theta 4-го луча", "AоD phi 5-го луча", "AоD theta 5-го луча", "AоD phi 6-го луча", "AоD theta 6-го луча", "AоD phi 7-го луча", "AоD theta 7-го луча", "AоD phi 8-го луча", "AоD theta 8-го луча",  "AоD phi 9-го луча", "AоD theta 9-го луча", "AоD phi 10-го луча", "AоD theta 10-го луча", "AоD phi 11-го луча", "AоD theta 11-го луча", "AоD phi 12-го луча", "AоD theta 12-го луча", "AоD phi 13-го луча", "AоD theta 13-го луча", "AоA phi 13-го луча", "AоD phi 14-го луча", "AоD theta 14-го луча", "AоD phi 15-го луча", "AоD theta 15-го луча","N1")]
        test_data = test_data.loc[:,("мощность 1-го луча", "мощность 2-го луча",  "мощность 3-го луча",  "мощность 4-го луча", "мощность 5-го луча","мощность 6-го луча",  "мощность 7-го луча", "мощность 8-го луча", "мощность 9-го луча", "мощность 10-го луча", "мощность 11-го луча", "мощность 12-го луча", "мощность 13-го луча", "мощность 14-го луча", "мощность 15-го луча","AоD phi 1-го луча", "AоD theta 1-го луча","AоD phi 2-го луча", "AоD theta 2-го луча","AоD phi 3-го луча", "AоD theta 3-го луча", "AоD phi 4-го луча", "AоD theta 4-го луча", "AоD phi 5-го луча", "AоD theta 5-го луча", "AоD phi 6-го луча", "AоD theta 6-го луча", "AоD phi 7-го луча", "AоD theta 7-го луча", "AоD phi 8-го луча", "AоD theta 8-го луча",  "AоD phi 9-го луча", "AоD theta 9-го луча", "AоD phi 10-го луча", "AоD theta 10-го луча", "AоD phi 11-го луча", "AоD theta 11-го луча", "AоD phi 12-го луча", "AоD theta 12-го луча", "AоD phi 13-го луча", "AоD theta 13-го луча", "AоA phi 13-го луча", "AоD phi 14-го луча", "AоD theta 14-го луча", "AоD phi 15-го луча", "AоD theta 15-го луча","N1")]
    
  
        # ВРЕМЯ И AOD
        # train_data = train_data.loc[:,("время 1-го луча","время 2-го луча","время 3-го луча","время 4-го луча","время 5-го луча","время 6-го луча","время 7-го луча","время 8-го луча","время 9-го луча","время 10-го луча","время 11-го луча","время 12-го луча","время 13-го луча","время 14-го луча","время 15-го луча","AоD phi 1-го луча", "AоD theta 1-го луча","AоD phi 2-го луча", "AоD theta 2-го луча","AоD phi 3-го луча", "AоD theta 3-го луча", "AоD phi 4-го луча", "AоD theta 4-го луча", "AоD phi 5-го луча", "AоD theta 5-го луча", "AоD phi 6-го луча", "AоD theta 6-го луча", "AоD phi 7-го луча", "AоD theta 7-го луча", "AоD phi 8-го луча", "AоD theta 8-го луча",  "AоD phi 9-го луча", "AоD theta 9-го луча", "AоD phi 10-го луча", "AоD theta 10-го луча", "AоD phi 11-го луча", "AоD theta 11-го луча", "AоD phi 12-го луча", "AоD theta 12-го луча", "AоD phi 13-го луча", "AоD theta 13-го луча", "AоD phi 13-го луча", "AоD phi 14-го луча", "AоD theta 14-го луча", "AоD phi 15-го луча", "AоD theta 15-го луча")]
        # test_data = test_data.loc[:,("время 1-го луча","время 2-го луча","время 3-го луча","время 4-го луча","время 5-го луча","время 6-го луча","время 7-го луча","время 8-го луча","время 9-го луча","время 10-го луча","время 11-го луча","время 12-го луча","время 13-го луча","время 14-го луча","время 15-го луча","AоD phi 1-го луча", "AоD theta 1-го луча","AоD phi 2-го луча", "AоD theta 2-го луча","AоD phi 3-го луча", "AоD theta 3-го луча", "AоD phi 4-го луча", "AоD theta 4-го луча", "AоD phi 5-го луча", "AоD theta 5-го луча", "AоD phi 6-го луча", "AоD theta 6-го луча", "AоD phi 7-го луча", "AоD theta 7-го луча", "AоD phi 8-го луча", "AоD theta 8-го луча",  "AоD phi 9-го луча", "AоD theta 9-го луча", "AоD phi 10-го луча", "AоD theta 10-го луча", "AоD phi 11-го луча", "AоD theta 11-го луча", "AоD phi 12-го луча", "AоD theta 12-го луча", "AоD phi 13-го луча", "AоD theta 13-го луча", "AоD phi 13-го луча", "AоD phi 14-го луча", "AоD theta 14-го луча", "AоD phi 15-го луча", "AоD theta 15-го луча")]


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
    
        X_MAE_AV_1.append(x_mae)
        X_MSE_AV_1.append(x_mse)
        Y_MAE_AV_1.append(y_mae)
        Y_MSE_AV_1.append(y_mse)
        x_test_1.extend(y_test_for_x)
        y_test_1.extend(y_test_for_y)
        x_predict_1.extend(X_predict_for_x)
        y_predict_1.extend(X_predict_for_y)
        y_error_1.extend(y_err)
            
        d = (x_err**2+y_err**2)**1/2
        print('d: %.3f' % (d))
        dir_1.extend(d)
    
        print(i)
        Results_1 = [X_MSE_AV_1,X_MAE_AV_1,Y_MSE_AV_1,Y_MAE_AV_1,x_test_1,y_test_1,dir_1]
        Results_1 = pd.DataFrame(Results_1)
        Results_1 = Results_1.transpose()
        Results_1 = round(Results_1,3)
        writer = pd.ExcelWriter('Results_1.xlsx')
        Results_1.to_excel(writer) 
        writer.save()


AV_X_MSE_1 = round(sum(X_MSE_AV_1) / (k+1),3)
AV_X_MAE_1 = round(sum(X_MAE_AV_1) / (k+1),3)
AV_Y_MSE_1 = round(sum(Y_MSE_AV_1) / (k+1),3)
AV_Y_MAE_1 = round(sum(Y_MAE_AV_1) / (k+1),3)
av_dir_1 = round(sum(dir_1) / (k+1),3)

print(AV_X_MSE_1)
print(AV_X_MAE_1)
print(AV_Y_MSE_1)
print(AV_Y_MAE_1)
print(av_dir_1)

Results_1.columns=["X_MSE_AV_1","X_MAE_AV_1","Y_MSE_AV_1","Y_MAE_AV_1","x_test_1","y_test_1","dir_1"]
Results_1.to_csv("Results_Tx_1.csv",encoding = 'cp1251')
df = Results_1[["x_test_1","y_test_1","dir_1"]]
df=df.pivot('x_test_1', 'y_test_1', 'dir_1')
ax = sns.heatmap(df,square=True,cmap = sns.cm.rocket_r)
plt.title('Геометрическое расстояние м/у предсказанной и измеренной точкой', fontsize=8)
plt.title('d')
plt.ylabel('x')
plt.xlabel('y')
plt.gca().invert_yaxis()
plt.show()

k=414
for i in range(0,k):
        test_data = df2.iloc[[i]]
        train_data  = df_Tx2.query("index != @i")
        train_data.to_csv("train_data.csv",encoding = 'cp1251')
        
        #отделить признаки от ярлыков
        train_labels = train_data.loc[:,('x','y')]
        test_labels = test_data.loc[:,('x','y')]
        
        #создание обучающих и тестовых наборов
        # test_data = df.sample(n=1, replace=True)
        # train_data = df.drop(test_data.index)
        # train_data.to_csv("train_data.csv",encoding = 'cp1251')
        
        
        #мощность и АОD
        train_data = train_data.loc[:,("мощность 1-го луча", "мощность 2-го луча", "мощность 3-го луча", "мощность 4-го луча", "мощность 5-го луча", "мощность 6-го луча",  "мощность 7-го луча", "мощность 8-го луча", "мощность 9-го луча", "мощность 10-го луча", "мощность 11-го луча", "мощность 12-го луча", "мощность 13-го луча", "мощность 14-го луча", "мощность 15-го луча","AоD phi 1-го луча", "AоD theta 1-го луча","AоD phi 2-го луча", "AоD theta 2-го луча","AоD phi 3-го луча", "AоD theta 3-го луча", "AоD phi 4-го луча", "AоD theta 4-го луча", "AоD phi 5-го луча", "AоD theta 5-го луча", "AоD phi 6-го луча", "AоD theta 6-го луча", "AоD phi 7-го луча", "AоD theta 7-го луча", "AоD phi 8-го луча", "AоD theta 8-го луча",  "AоD phi 9-го луча", "AоD theta 9-го луча", "AоD phi 10-го луча", "AоD theta 10-го луча", "AоD phi 11-го луча", "AоD theta 11-го луча", "AоD phi 12-го луча", "AоD theta 12-го луча", "AоD phi 13-го луча", "AоD theta 13-го луча", "AоA phi 13-го луча", "AоD phi 14-го луча", "AоD theta 14-го луча", "AоD phi 15-го луча", "AоD theta 15-го луча","N2")]
        test_data = test_data.loc[:,("мощность 1-го луча", "мощность 2-го луча",  "мощность 3-го луча",  "мощность 4-го луча", "мощность 5-го луча","мощность 6-го луча",  "мощность 7-го луча", "мощность 8-го луча", "мощность 9-го луча", "мощность 10-го луча", "мощность 11-го луча", "мощность 12-го луча", "мощность 13-го луча", "мощность 14-го луча", "мощность 15-го луча","AоD phi 1-го луча", "AоD theta 1-го луча","AоD phi 2-го луча", "AоD theta 2-го луча","AоD phi 3-го луча", "AоD theta 3-го луча", "AоD phi 4-го луча", "AоD theta 4-го луча", "AоD phi 5-го луча", "AоD theta 5-го луча", "AоD phi 6-го луча", "AоD theta 6-го луча", "AоD phi 7-го луча", "AоD theta 7-го луча", "AоD phi 8-го луча", "AоD theta 8-го луча",  "AоD phi 9-го луча", "AоD theta 9-го луча", "AоD phi 10-го луча", "AоD theta 10-го луча", "AоD phi 11-го луча", "AоD theta 11-го луча", "AоD phi 12-го луча", "AоD theta 12-го луча", "AоD phi 13-го луча", "AоD theta 13-го луча", "AоA phi 13-го луча", "AоD phi 14-го луча", "AоD theta 14-го луча", "AоD phi 15-го луча", "AоD theta 15-го луча","N2")]
    
  
        # ВРЕМЯ И AOD
        # train_data = train_data.loc[:,("время 1-го луча","время 2-го луча","время 3-го луча","время 4-го луча","время 5-го луча","время 6-го луча","время 7-го луча","время 8-го луча","время 9-го луча","время 10-го луча","время 11-го луча","время 12-го луча","время 13-го луча","время 14-го луча","время 15-го луча","AоD phi 1-го луча", "AоD theta 1-го луча","AоD phi 2-го луча", "AоD theta 2-го луча","AоD phi 3-го луча", "AоD theta 3-го луча", "AоD phi 4-го луча", "AоD theta 4-го луча", "AоD phi 5-го луча", "AоD theta 5-го луча", "AоD phi 6-го луча", "AоD theta 6-го луча", "AоD phi 7-го луча", "AоD theta 7-го луча", "AоD phi 8-го луча", "AоD theta 8-го луча",  "AоD phi 9-го луча", "AоD theta 9-го луча", "AоD phi 10-го луча", "AоD theta 10-го луча", "AоD phi 11-го луча", "AоD theta 11-го луча", "AоD phi 12-го луча", "AоD theta 12-го луча", "AоD phi 13-го луча", "AоD theta 13-го луча", "AоD phi 13-го луча", "AоD phi 14-го луча", "AоD theta 14-го луча", "AоD phi 15-го луча", "AоD theta 15-го луча")]
        # test_data = test_data.loc[:,("время 1-го луча","время 2-го луча","время 3-го луча","время 4-го луча","время 5-го луча","время 6-го луча","время 7-го луча","время 8-го луча","время 9-го луча","время 10-го луча","время 11-го луча","время 12-го луча","время 13-го луча","время 14-го луча","время 15-го луча","AоD phi 1-го луча", "AоD theta 1-го луча","AоD phi 2-го луча", "AоD theta 2-го луча","AоD phi 3-го луча", "AоD theta 3-го луча", "AоD phi 4-го луча", "AоD theta 4-го луча", "AоD phi 5-го луча", "AоD theta 5-го луча", "AоD phi 6-го луча", "AоD theta 6-го луча", "AоD phi 7-го луча", "AоD theta 7-го луча", "AоD phi 8-го луча", "AоD theta 8-го луча",  "AоD phi 9-го луча", "AоD theta 9-го луча", "AоD phi 10-го луча", "AоD theta 10-го луча", "AоD phi 11-го луча", "AоD theta 11-го луча", "AоD phi 12-го луча", "AоD theta 12-го луча", "AоD phi 13-го луча", "AоD theta 13-го луча", "AоD phi 13-го луча", "AоD phi 14-го луча", "AоD theta 14-го луча", "AоD phi 15-го луча", "AоD theta 15-го луча")]


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
    
        X_MAE_AV_2.append(x_mae)
        X_MSE_AV_2.append(x_mse)
        Y_MAE_AV_2.append(y_mae)
        Y_MSE_AV_2.append(y_mse)
        x_test_2.extend(y_test_for_x)
        y_test_2.extend(y_test_for_y)
        x_predict_2.extend(X_predict_for_x)
        y_predict_2.extend(X_predict_for_y)
        y_error_2.extend(y_err)
            
        d = (x_err**2+y_err**2)**1/2
        print('d: %.3f' % (d))
        dir_2.extend(d)
    
        print(i)
        Results_2 = [X_MSE_AV_2,X_MAE_AV_2,Y_MSE_AV_2,Y_MAE_AV_2,x_test_2,y_test_2,dir_2]
        Results_2 = pd.DataFrame(Results_1)
        Results_2 = Results_1.transpose()
        Results_2 = round(Results_1,3)
        writer = pd.ExcelWriter('Results_2.xlsx')
        Results_2.to_excel(writer) 
        writer.save()


AV_X_MSE_2 = round(sum(X_MSE_AV_2) / (k+1),3)
AV_X_MAE_2 = round(sum(X_MAE_AV_2) / (k+1),3)
AV_Y_MSE_2 = round(sum(Y_MSE_AV_2) / (k+1),3)
AV_Y_MAE_2 = round(sum(Y_MAE_AV_2) / (k+1),3)
av_dir_2 = round(sum(dir_2) / (k+1),3)

print(AV_X_MSE_1)
print(AV_X_MAE_1)
print(AV_Y_MSE_1)
print(AV_Y_MAE_1)
print(av_dir_1)

Results_2.columns=["X_MSE_AV_2","X_MAE_AV_2","Y_MSE_AV_2","Y_MAE_AV_2","x_test_2","y_test_2","dir_2"]
Results_2.to_csv("Results_Tx_2.csv",encoding = 'cp1251')
df = Results_2[["x_test_2","y_test_2","dir_2"]]
df=df.pivot('x_test_2', 'y_test_2', 'dir_2')
ax = sns.heatmap(df,square=True,cmap = sns.cm.rocket_r)
plt.title('Геометрическое расстояние м/у предсказанной и измеренной точкой', fontsize=8)
plt.title('d')
plt.ylabel('x')
plt.xlabel('y')
plt.gca().invert_yaxis()
plt.show()

k=414
for i in range(0,k):
        test_data = df3.iloc[[i]]
        train_data  = df_Tx3.query("index != @i")
        train_data.to_csv("train_data.csv",encoding = 'cp1251')
        
        #отделить признаки от ярлыков
        train_labels = train_data.loc[:,('x','y')]
        test_labels = test_data.loc[:,('x','y')]
        
        #создание обучающих и тестовых наборов
        # test_data = df.sample(n=1, replace=True)
        # train_data = df.drop(test_data.index)
        # train_data.to_csv("train_data.csv",encoding = 'cp1251')
        
        
        #мощность и АОD
        train_data = train_data.loc[:,("мощность 1-го луча", "мощность 2-го луча", "мощность 3-го луча", "мощность 4-го луча", "мощность 5-го луча", "мощность 6-го луча",  "мощность 7-го луча", "мощность 8-го луча", "мощность 9-го луча", "мощность 10-го луча", "мощность 11-го луча", "мощность 12-го луча", "мощность 13-го луча", "мощность 14-го луча", "мощность 15-го луча","AоD phi 1-го луча", "AоD theta 1-го луча","AоD phi 2-го луча", "AоD theta 2-го луча","AоD phi 3-го луча", "AоD theta 3-го луча", "AоD phi 4-го луча", "AоD theta 4-го луча", "AоD phi 5-го луча", "AоD theta 5-го луча", "AоD phi 6-го луча", "AоD theta 6-го луча", "AоD phi 7-го луча", "AоD theta 7-го луча", "AоD phi 8-го луча", "AоD theta 8-го луча",  "AоD phi 9-го луча", "AоD theta 9-го луча", "AоD phi 10-го луча", "AоD theta 10-го луча", "AоD phi 11-го луча", "AоD theta 11-го луча", "AоD phi 12-го луча", "AоD theta 12-го луча", "AоD phi 13-го луча", "AоD theta 13-го луча", "AоA phi 13-го луча", "AоD phi 14-го луча", "AоD theta 14-го луча", "AоD phi 15-го луча", "AоD theta 15-го луча","N3")]
        test_data = test_data.loc[:,("мощность 1-го луча", "мощность 2-го луча",  "мощность 3-го луча",  "мощность 4-го луча", "мощность 5-го луча","мощность 6-го луча",  "мощность 7-го луча", "мощность 8-го луча", "мощность 9-го луча", "мощность 10-го луча", "мощность 11-го луча", "мощность 12-го луча", "мощность 13-го луча", "мощность 14-го луча", "мощность 15-го луча","AоD phi 1-го луча", "AоD theta 1-го луча","AоD phi 2-го луча", "AоD theta 2-го луча","AоD phi 3-го луча", "AоD theta 3-го луча", "AоD phi 4-го луча", "AоD theta 4-го луча", "AоD phi 5-го луча", "AоD theta 5-го луча", "AоD phi 6-го луча", "AоD theta 6-го луча", "AоD phi 7-го луча", "AоD theta 7-го луча", "AоD phi 8-го луча", "AоD theta 8-го луча",  "AоD phi 9-го луча", "AоD theta 9-го луча", "AоD phi 10-го луча", "AоD theta 10-го луча", "AоD phi 11-го луча", "AоD theta 11-го луча", "AоD phi 12-го луча", "AоD theta 12-го луча", "AоD phi 13-го луча", "AоD theta 13-го луча", "AоA phi 13-го луча", "AоD phi 14-го луча", "AоD theta 14-го луча", "AоD phi 15-го луча", "AоD theta 15-го луча","N3")]
    
  
        # ВРЕМЯ И AOD
        # train_data = train_data.loc[:,("время 1-го луча","время 2-го луча","время 3-го луча","время 4-го луча","время 5-го луча","время 6-го луча","время 7-го луча","время 8-го луча","время 9-го луча","время 10-го луча","время 11-го луча","время 12-го луча","время 13-го луча","время 14-го луча","время 15-го луча","AоD phi 1-го луча", "AоD theta 1-го луча","AоD phi 2-го луча", "AоD theta 2-го луча","AоD phi 3-го луча", "AоD theta 3-го луча", "AоD phi 4-го луча", "AоD theta 4-го луча", "AоD phi 5-го луча", "AоD theta 5-го луча", "AоD phi 6-го луча", "AоD theta 6-го луча", "AоD phi 7-го луча", "AоD theta 7-го луча", "AоD phi 8-го луча", "AоD theta 8-го луча",  "AоD phi 9-го луча", "AоD theta 9-го луча", "AоD phi 10-го луча", "AоD theta 10-го луча", "AоD phi 11-го луча", "AоD theta 11-го луча", "AоD phi 12-го луча", "AоD theta 12-го луча", "AоD phi 13-го луча", "AоD theta 13-го луча", "AоD phi 13-го луча", "AоD phi 14-го луча", "AоD theta 14-го луча", "AоD phi 15-го луча", "AоD theta 15-го луча")]
        # test_data = test_data.loc[:,("время 1-го луча","время 2-го луча","время 3-го луча","время 4-го луча","время 5-го луча","время 6-го луча","время 7-го луча","время 8-го луча","время 9-го луча","время 10-го луча","время 11-го луча","время 12-го луча","время 13-го луча","время 14-го луча","время 15-го луча","AоD phi 1-го луча", "AоD theta 1-го луча","AоD phi 2-го луча", "AоD theta 2-го луча","AоD phi 3-го луча", "AоD theta 3-го луча", "AоD phi 4-го луча", "AоD theta 4-го луча", "AоD phi 5-го луча", "AоD theta 5-го луча", "AоD phi 6-го луча", "AоD theta 6-го луча", "AоD phi 7-го луча", "AоD theta 7-го луча", "AоD phi 8-го луча", "AоD theta 8-го луча",  "AоD phi 9-го луча", "AоD theta 9-го луча", "AоD phi 10-го луча", "AоD theta 10-го луча", "AоD phi 11-го луча", "AоD theta 11-го луча", "AоD phi 12-го луча", "AоD theta 12-го луча", "AоD phi 13-го луча", "AоD theta 13-го луча", "AоD phi 13-го луча", "AоD phi 14-го луча", "AоD theta 14-го луча", "AоD phi 15-го луча", "AоD theta 15-го луча")]


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
    
        X_MAE_AV_3.append(x_mae)
        X_MSE_AV_3.append(x_mse)
        Y_MAE_AV_3.append(y_mae)
        Y_MSE_AV_3.append(y_mse)
        x_test_3.extend(y_test_for_x)
        y_test_3.extend(y_test_for_y)
        x_predict_3.extend(X_predict_for_x)
        y_predict_3.extend(X_predict_for_y)
        y_error_3.extend(y_err)
            
        d = (x_err**2+y_err**2)**1/2
        print('d: %.3f' % (d))
        dir_3.extend(d)
    
        print(i)
        Results_3 = [X_MSE_AV_3,X_MAE_AV_3,Y_MSE_AV_3,Y_MAE_AV_3,x_test_3,y_test_3,dir_3]
        Results_3 = pd.DataFrame(Results_3)
        Results_3 = Results_3.transpose()
        Results_3 = round(Results_3,3)
        writer = pd.ExcelWriter('Results_1.xlsx')
        Results_3.to_excel(writer) 
        writer.save()


AV_X_MSE_3 = round(sum(X_MSE_AV_3) / (k+1),3)
AV_X_MAE_3 = round(sum(X_MAE_AV_3) / (k+1),3)
AV_Y_MSE_3 = round(sum(Y_MSE_AV_3) / (k+1),3)
AV_Y_MAE_3 = round(sum(Y_MAE_AV_3) / (k+1),3)
av_dir_3 = round(sum(dir_3) / (k+1),3)

print(AV_X_MSE_3)
print(AV_X_MAE_3)
print(AV_Y_MSE_3)
print(AV_Y_MAE_3)
print(av_dir_3)

Results_3.columns=["X_MSE_AV_3","X_MAE_AV_3","Y_MSE_AV_3","Y_MAE_AV_3","x_test_3","y_test_3","dir_3"]
Results_3.to_csv("Results_Tx_3.csv",encoding = 'cp1251')
df = Results_3[["x_test_3","y_test_3","dir_3"]]
df=df.pivot('x_test_3', 'y_test_3', 'dir_3')
ax = sns.heatmap(df,square=True,cmap = sns.cm.rocket_r)
plt.title('Геометрическое расстояние м/у предсказанной и измеренной точкой', fontsize=8)
plt.title('d')
plt.ylabel('x')
plt.xlabel('y')
plt.gca().invert_yaxis()
plt.show()


print((AV_X_MSE_1+AV_X_MSE_2+AV_X_MSE_3)/3)
print((AV_X_MAE_1+AV_X_MAE_2+AV_X_MAE_3)/3)
print((AV_Y_MSE_1+AV_Y_MSE_2+AV_Y_MSE_3)/3)
print((AV_Y_MAE_1+AV_Y_MAE_2+AV_Y_MAE_3)/3)
print((av_dir_1+av_dir_2+av_dir_3)/3)