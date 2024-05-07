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
from mpl_toolkits import mplot3d
#открываем рабочий каталог ('cwd')
cwd=os.getcwd()
cwd

#изменение директории
os.chdir("C:\\Users\\Азалия\\Desktop\\Диплом\\Начальные данные\\Моя версия данных")

#информация об именах файлов
os.listdir('.')
#создание серии названий параметров
series1 = pd.Series(["фаза 1-го луча", "мощность 1-го луча", "время 1-го луча", "AоD phi 1-го луча", "AоD theta 1-го луча", "AоA phi 1-го луча", "AоА theta 1-го луча", "фаза 2-го луча", "мощность 2-го луча", "время 2-го луча", "AоD phi 2-го луча", "AоD theta 2-го луча", "AоA phi 2-го луча", "AоА theta 2-го луча", "фаза 3-го луча", "мощность 3-го луча", "время 3-го луча", "AоD phi 3-го луча", "AоD theta 3-го луча", "AоA phi 3-го луча", "AоА theta 3-го луча", "фаза 4-го луча", "мощность 4-го луча", "время 4-го луча", "AоD phi 4-го луча", "AоD theta 4-го луча", "AоA phi 4-го луча", "AоА theta 4-го луча", "фаза 5-го луча", "мощность 5-го луча", "время 5-го луча", "AоD phi 5-го луча", "AоD theta 5-го луча", "AоA phi 5-го луча", "AоА theta 5-го луча", "фаза 6-го луча", "мощность 6-го луча", "время 6-го луча", "AоD phi 6-го луча", "AоD theta 6-го луча", "AоA phi 6-го луча", "AоА theta 6-го луча", "фаза 7-го луча", "мощность 7-го луча", "время 7-го луча", "AоD phi 7-го луча", "AоD theta 7-го луча", "AоA phi 7-го луча", "AоА theta 7-го луча", "фаза 8-го луча", "мощность 8-го луча", "время 8-го луча", "AоD phi 8-го луча", "AоD theta 8-го луча", "AоA phi 8-го луча", "AоА theta 8-го луча", "фаза 9-го луча", "мощность 9-го луча", "время 9-го луча", "AоD phi 9-го луча", "AоD theta 9-го луча", "AоA phi 9-го луча", "AоА theta 9-го луча", "фаза 10-го луча", "мощность 10-го луча", "время 10-го луча", "AоD phi 10-го луча", "AоD theta 10-го луча", "AоA phi 10-го луча", "AоА theta 10-го луча", "фаза 11-го луча", "мощность 11-го луча", "время 11-го луча", "AоD phi 11-го луча", "AоD theta 11-го луча", "AоA phi 11-го луча", "AоА theta 11-го луча", "фаза 12-го луча", "мощность 12-го луча", "время 12-го луча", "AоD phi 12-го луча", "AоD theta 12-го луча", "AоA phi 12-го луча", "AоА theta 12-го луча", "фаза 13-го луча", "мощность 13-го луча", "время 13-го луча", "AоD phi 13-го луча", "AоD theta 13-го луча", "AоA phi 13-го луча", "AоА theta 13-го луча", "фаза 14-го луча", "мощность 14-го луча", "время 14-го луча", "AоD phi 14-го луча", "AоD theta 14-го луча", "AоA phi 14-го луча", "AоА theta 14-го луча", "фаза 15-го луча", "мощность 15-го луча", "время 15-го луча", "AоD phi 15-го луча", "AоD theta 15-го луча", "AоA phi 15-го луча", "AоА theta 15-го луча",'x','y','z'])

#создание структурированной таблицы данных                     
# df = pd.read_csv("DATA_FULL_Tx1_WALLS_EMPTY.csv",header=None, names=series1)
df1 = pd.read_csv("DATA_FULL_Tx1_WALLS_OBTACLES.csv",header=None, names=series1)
df2 = pd.read_csv("DATA_FULL_Tx2_WALLS_OBTACLES.csv", names=series1)
df3 = pd.read_csv("DATA_FULL_Tx3_WALLS_OBTACLES.csv", names=series1)

# объединение данных по трем точкам доступа в одну
df = pd.concat([df1, df2,df3])
df = round(df,3)

# print(df)

#удаление точек в стенах
df = df.drop(df[(df.x==0)].index)
df = df.drop(df[(df.x==16)].index)
df = df.drop(df[(df.y==0)].index)
df = df.drop(df[(df.y==24)].index)
df = df.drop(df[(df.y==32)].index)
df = df.drop(df[(df.x==6)&(df.y<=24)].index)
df = df.drop(df[(df.x==10)&(df.y<=24)].index)
df = df.drop(df[((df.x<=6)|((df.x>=10)&(df.x<=16)))&((df.y==6)|(df.y==12)|(df.y==18))].index)

#создание номеров комнат
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
    
# добавление номеров комнат в данные
x = df.loc[:,'x']
y = df.loc[:,'y']
N = list(map(even_odd,x,y))
df.loc[:, "N"] = N
# print(df.shape)

# исследование точности от количества комнат
# нижние 4 комнаты и часть коридора рядом с ними
# df = df.drop(df[(df.y>12)].index)
# нижние 6 комнат и часть коридора рядом с ними
# df = df.drop(df[(df.y>18)].index)
# нижние 8 комнат и часть коридора рядом с ними
# df = df.drop(df[(df.y>24)].index)

X_MSE_AV=[]
X_MAE_AV=[]
Y_MSE_AV=[]
Y_MAE_AV=[]
x_test=[]
y_test=[]

for i in range(1,100):
    #создание обучающих и тестовых наборов
    # train_data = df.sample(frac = 0.9,replace=True)
    # train_data.to_csv("train_data.csv",encoding = 'cp1251')
    # test_data = df.drop(train_data.index)
    test_data = df.sample(n=75, replace=True)
    train_data = df.drop(test_data.index)
    train_data.to_csv("train_data.csv",encoding = 'cp1251')
    
    # test_data = df.sample(n=1, replace=True)
    # train_data = df.drop(test_data.index)
    # train_data.to_csv("train_data.csv",encoding = 'cp1251')
    
    #отделить признаки от ярлыков
    train_labels = train_data.loc[:,('x','y')]
    test_labels = test_data.loc[:,('x','y')]
    
    # new_test_data = test_labels.transpose()
    # new_test_data.to_excel('./new_test_data.xlsx')
    
    
    # использование разных метрик локализации для обучения нейронной сети
    #мощность и АОD
    # train_data = train_data.loc[:,("мощность 1-го луча", "мощность 2-го луча", "мощность 3-го луча", "мощность 4-го луча", "мощность 5-го луча", "мощность 6-го луча",  "мощность 7-го луча", "мощность 8-го луча", "мощность 9-го луча", "мощность 10-го луча", "мощность 11-го луча", "мощность 12-го луча", "мощность 13-го луча", "мощность 14-го луча", "мощность 15-го луча","AоD phi 1-го луча", "AоD theta 1-го луча","AоD phi 2-го луча", "AоD theta 2-го луча","AоD phi 3-го луча", "AоD theta 3-го луча", "AоD phi 4-го луча", "AоD theta 4-го луча", "AоD phi 5-го луча", "AоD theta 5-го луча", "AоD phi 6-го луча", "AоD theta 6-го луча", "AоD phi 7-го луча", "AоD theta 7-го луча", "AоD phi 8-го луча", "AоD theta 8-го луча",  "AоD phi 9-го луча", "AоD theta 9-го луча", "AоD phi 10-го луча", "AоD theta 10-го луча", "AоD phi 11-го луча", "AоD theta 11-го луча", "AоD phi 12-го луча", "AоD theta 12-го луча", "AоD phi 13-го луча", "AоD theta 13-го луча", "AоA phi 13-го луча", "AоD phi 14-го луча", "AоD theta 14-го луча", "AоD phi 15-го луча", "AоD theta 15-го луча")]
    # test_data = test_data.loc[:,("мощность 1-го луча", "мощность 2-го луча",  "мощность 3-го луча",  "мощность 4-го луча", "мощность 5-го луча","мощность 6-го луча",  "мощность 7-го луча", "мощность 8-го луча", "мощность 9-го луча", "мощность 10-го луча", "мощность 11-го луча", "мощность 12-го луча", "мощность 13-го луча", "мощность 14-го луча", "мощность 15-го луча","AоD phi 1-го луча", "AоD theta 1-го луча","AоD phi 2-го луча", "AоD theta 2-го луча","AоD phi 3-го луча", "AоD theta 3-го луча", "AоD phi 4-го луча", "AоD theta 4-го луча", "AоD phi 5-го луча", "AоD theta 5-го луча", "AоD phi 6-го луча", "AоD theta 6-го луча", "AоD phi 7-го луча", "AоD theta 7-го луча", "AоD phi 8-го луча", "AоD theta 8-го луча",  "AоD phi 9-го луча", "AоD theta 9-го луча", "AоD phi 10-го луча", "AоD theta 10-го луча", "AоD phi 11-го луча", "AоD theta 11-го луча", "AоD phi 12-го луча", "AоD theta 12-го луча", "AоD phi 13-го луча", "AоD theta 13-го луча", "AоA phi 13-го луча", "AоD phi 14-го луча", "AоD theta 14-го луча", "AоD phi 15-го луча", "AоD theta 15-го луча")]
    
    
    # проверка нормализации (все параметры должны быть распределены в пределах [-1;1])
    # norm_train_data = norm(train_data)
    # norm_train_data = norm_train_data.describe()
    # norm_train_data = norm_train_data.transpose()
    # print('Максимальные значения нормированных параметров:')
    # print(norm_train_data['max'])
    # print('Минимальные значения нормированных параметров:')
    # print(norm_train_data['min'])


    train_data = train_data.loc[:,("мощность 1-го луча", "мощность 2-го луча", "мощность 3-го луча", "мощность 4-го луча", "мощность 5-го луча", "мощность 6-го луча",  "мощность 7-го луча", "мощность 8-го луча", "мощность 9-го луча", "мощность 10-го луча", "мощность 11-го луча", "мощность 12-го луча", "мощность 13-го луча", "мощность 14-го луча", "мощность 15-го луча","AоD phi 1-го луча", "AоD theta 1-го луча","AоD phi 2-го луча", "AоD theta 2-го луча","AоD phi 3-го луча", "AоD theta 3-го луча", "AоD phi 4-го луча", "AоD theta 4-го луча", "AоD phi 5-го луча", "AоD theta 5-го луча", "AоD phi 6-го луча", "AоD theta 6-го луча", "AоD phi 7-го луча", "AоD theta 7-го луча", "AоD phi 8-го луча", "AоD theta 8-го луча",  "AоD phi 9-го луча", "AоD theta 9-го луча", "AоD phi 10-го луча", "AоD theta 10-го луча", "AоD phi 11-го луча", "AоD theta 11-го луча", "AоD phi 12-го луча", "AоD theta 12-го луча", "AоD phi 13-го луча", "AоD theta 13-го луча", "AоA phi 13-го луча", "AоD phi 14-го луча", "AоD theta 14-го луча", "AоD phi 15-го луча", "AоD theta 15-го луча","N")]
    test_data = test_data.loc[:,("мощность 1-го луча", "мощность 2-го луча",  "мощность 3-го луча",  "мощность 4-го луча", "мощность 5-го луча","мощность 6-го луча",  "мощность 7-го луча", "мощность 8-го луча", "мощность 9-го луча", "мощность 10-го луча", "мощность 11-го луча", "мощность 12-го луча", "мощность 13-го луча", "мощность 14-го луча", "мощность 15-го луча","AоD phi 1-го луча", "AоD theta 1-го луча","AоD phi 2-го луча", "AоD theta 2-го луча","AоD phi 3-го луча", "AоD theta 3-го луча", "AоD phi 4-го луча", "AоD theta 4-го луча", "AоD phi 5-го луча", "AоD theta 5-го луча", "AоD phi 6-го луча", "AоD theta 6-го луча", "AоD phi 7-го луча", "AоD theta 7-го луча", "AоD phi 8-го луча", "AоD theta 8-го луча",  "AоD phi 9-го луча", "AоD theta 9-го луча", "AоD phi 10-го луча", "AоD theta 10-го луча", "AоD phi 11-го луча", "AоD theta 11-го луча", "AоD phi 12-го луча", "AоD theta 12-го луча", "AоD phi 13-го луча", "AоD theta 13-го луча", "AоA phi 13-го луча", "AоD phi 14-го луча", "AоD theta 14-го луча", "AоD phi 15-го луча", "AоD theta 15-го луча","N")]

    train_stats = train_data.describe()
    train_stats = train_stats.transpose()

    #нормализация входных данных
    def norm(x):
      return (x - train_stats['min']) /  (train_stats['max'] - train_stats['min'])
    
    inputs = keras.Input(shape=(len(train_data.keys())), name="digits")
    x = layers.Dense(125, activation="relu", name="dense_1")(inputs)
    x = layers.Dense(125, activation="relu", name="dense_2")(x)
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
      validation_split = 0.2,
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
    
    # d = (x_err**2+y_err**2)**1/2
    # y_test_for_x = pd.Series(y_test_for_x)
    # X_predict_for_x = pd.Series(X_predict_for_x)
    # y_test_for_y = pd.Series(y_test_for_y)
    # X_predict_for_y = pd.Series(X_predict_for_y)
    # d = pd.Series(d)
    # massiv = pd.DataFrame({'x': y_test_for_x,'y': y_test_for_y,'d': d})
    # # график x_err и y_err в зависимости от x и y
    # plt.figure()
    # plt.grid(True)
    # plt.scatter(y_test_for_x,X_predict_for_x)
    # plt.scatter(y_test_for_y,X_predict_for_y)
    # plt.ylabel('x_err or y_err')
    # plt.xlabel('x or y')
    # plt.legend(['x_err(x)', 'y_err(y)'], loc='upper left')
    # plt.show()
    # # тепловая карта
    # D = d.to_numpy()
    # X = massiv.loc[:,'x']
    # X = X.to_numpy()
    # Y = massiv.loc[:,'y']
    # Y = Y.to_numpy()
    # array = np.zeros((17,33))
    # array[X,Y] = D
    # array = np.transpose(array)
    # # print(array)
    # sns.heatmap(array,square=True,cmap = sns.cm.rocket_r)
    # # plt.title('Геометрическое расстояние м/у предсказанной и измеренной точкой', fontsize=8)
    # plt.title('d')
    # plt.ylabel('y')
    # plt.xlabel('x')
    # plt.gca().invert_yaxis()
    # plt.show()
    #трёхмерный график
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # ax.scatter(X, Y, D,c = 'g')
    # ax.set_xlabel('x')
    # ax.set_ylabel('y')
    # ax.set_zlabel('d')
    # ax.set_title('Геометрическое расстояние м/у предсказанной и\nизмеренной точкой в зависимости от x и у')
    # plt.grid(True)
    # # положение камеры - высота и угол наклона очей в градусах
    # ax.view_init(15, 270)
    # plt.show() 
    # print(test_labels)
    # print(test_labels.shape)
    print(i)
    Results = [X_MSE_AV,X_MAE_AV,Y_MSE_AV,Y_MAE_AV,x_test,y_test]
    Results = pd.DataFrame(Results)
    Results = Results.transpose()
    Results = round(Results,3)
    writer = pd.ExcelWriter('Results.xlsx')
    Results.to_excel(writer) 
    writer.save()



AV_X_MSE = round(sum(X_MSE_AV) / 100,3)
AV_X_MAE = round(sum(X_MAE_AV) / 100,3)
AV_Y_MSE = round(sum(Y_MSE_AV) / 100,3)
AV_Y_MAE = round(sum(Y_MAE_AV) / 100,3)
print(AV_X_MSE)
print(AV_X_MAE)
print(AV_Y_MSE)
print(AV_Y_MAE)
