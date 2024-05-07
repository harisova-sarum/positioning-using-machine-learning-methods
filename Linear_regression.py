import os
#Retrieve current working directory ('cwd')
cwd=os.getcwd()
cwd
#Change directory
os.chdir("D:\\1 семестр\\НИР. Завьялов\\Машинное обучение\\1 задание")
#List all files and directories in current direcctory
os.listdir('.')

import pandas as pd
series1 = pd.Series(["фаза 1-го луча", "мощность 1-го луча", "время 1-го луча", "AоD phi 1-го луча", "AоD theta 1-го луча", "AоA phi 1-го луча", "AоА theta 1-го луча", "фаза 2-го луча", "мощность 2-го луча", "время 2-го луча", "AоD phi 2-го луча", "AоD theta 2-го луча", "AоA phi 2-го луча", "AоА theta 2-го луча", "фаза 3-го луча", "мощность 3-го луча", "время 3-го луча", "AоD phi 3-го луча", "AоD theta 3-го луча", "AоA phi 3-го луча", "AоА theta 3-го луча", "фаза 4-го луча", "мощность 4-го луча", "время 4-го луча", "AоD phi 4-го луча", "AоD theta 4-го луча", "AоA phi 4-го луча", "AоА theta 4-го луча", "фаза 5-го луча", "мощность 5-го луча", "время 5-го луча", "AоD phi 5-го луча", "AоD theta 5-го луча", "AоA phi 5-го луча", "AоА theta 5-го луча", "фаза 6-го луча", "мощность 6-го луча", "время 6-го луча", "AоD phi 6-го луча", "AоD theta 6-го луча", "AоA phi 6-го луча", "AоА theta 6-го луча", "фаза 7-го луча", "мощность 7-го луча", "время 7-го луча", "AоD phi 7-го луча", "AоD theta 7-го луча", "AоA phi 7-го луча", "AоА theta 7-го луча", "фаза 8-го луча", "мощность 8-го луча", "время 8-го луча", "AоD phi 8-го луча", "AоD theta 8-го луча", "AоA phi 8-го луча", "AоА theta 8-го луча", "фаза 9-го луча", "мощность 9-го луча", "время 9-го луча", "AоD phi 9-го луча", "AоD theta 9-го луча", "AоA phi 9-го луча", "AоА theta 9-го луча", "фаза 10-го луча", "мощность 10-го луча", "время 10-го луча", "AоD phi 10-го луча", "AоD theta 10-го луча", "AоA phi 10-го луча", "AоА theta 10-го луча", "фаза 11-го луча", "мощность 11-го луча", "время 11-го луча", "AоD phi 11-го луча", "AоD theta 11-го луча", "AоA phi 11-го луча", "AоА theta 11-го луча", "фаза 12-го луча", "мощность 12-го луча", "время 12-го луча", "AоD phi 12-го луча", "AоD theta 12-го луча", "AоA phi 12-го луча", "AоА theta 12-го луча", "фаза 13-го луча", "мощность 13-го луча", "время 13-го луча", "AоD phi 13-го луча", "AоD theta 13-го луча", "AоA phi 13-го луча", "AоА theta 13-го луча", "фаза 14-го луча", "мощность 14-го луча", "время 14-го луча", "AоD phi 14-го луча", "AоD theta 14-го луча", "AоA phi 14-го луча", "AоА theta 14-го луча", "фаза 15-го луча", "мощность 15-го луча", "время 15-го луча", "AоD phi 15-го луча", "AоD theta 15-го луча", "AоA phi 15-го луча", "AоА theta 15-го луча",'x','y','z'])
#creating a structured data table                     
data = pd.read_csv("DATA_FULL_Tx1_WALLS_EMPTY.csv",header=None, names=series1)
#leaving the necessary columns
df = data[["мощность 1-го луча","мощность 2-го луча","мощность 3-го луча","мощность 4-го луча","мощность 5-го луча","мощность 6-го луча","мощность 7-го луча","мощность 8-го луча","мощность 9-го луча","мощность 10-го луча","мощность 11-го луча","мощность 12-го луча","мощность 13-го луча","мощность 14-го луча","мощность 15-го луча",'x','y']]
#deleting unnecessary coordinates
df = df.drop(df[(df.x==0)].index)
df = df.drop(df[(df.x==16)].index)
df = df.drop(df[(df.y==0)].index)
df = df.drop(df[(df.y==24)].index)
df = df.drop(df[(df.y==32)].index)
df = df.drop(df[(df.x==6)&(df.y<=24)].index)
df = df.drop(df[(df.x==10)&(df.y<=24)].index)
df = df.drop(df[((df.x<=6)|((df.x>=10)&(df.x<=16)))&((df.y==6)|(df.y==12)|(df.y==18))].index)


#use coding with russian symbols
df.to_csv("New_Data.csv",encoding = 'cp1251')
print("Форма массива: {}".format(df.shape))

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
# create training and test kits


import seaborn as sns
from sklearn import metrics
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
import pandas as pd
# df without coordinates

labels = df.loc[:,('x','y')]
data = df.loc[:,("мощность 1-го луча", "мощность 2-го луча", "мощность 3-го луча", "мощность 4-го луча", "мощность 5-го луча", "мощность 6-го луча",  "мощность 7-го луча", "мощность 8-го луча", "мощность 9-го луча", "мощность 10-го луча", "мощность 11-го луча", "мощность 12-го луча", "мощность 13-го луча", "мощность 14-го луча", "мощность 15-го луча")]

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
    #создание обучающих и тестовых наборов
    
    # train_data = df.sample(n=373)
    # train_data.to_csv("train_data.csv",encoding = 'cp1251')
    # test_data = df.drop(train_data.index)
    
    #отделить признаки от ярлыков
    train_labels = train_data.loc[:,('x','y')]
    test_labels = test_data.loc[:,('x','y')]
    
    train_data = train_data.loc[:,("мощность 1-го луча", "мощность 2-го луча", "мощность 3-го луча", "мощность 4-го луча", "мощность 5-го луча", "мощность 6-го луча",  "мощность 7-го луча", "мощность 8-го луча", "мощность 9-го луча", "мощность 10-го луча", "мощность 11-го луча", "мощность 12-го луча", "мощность 13-го луча", "мощность 14-го луча", "мощность 15-го луча")]
    test_data = test_data.loc[:,("мощность 1-го луча", "мощность 2-го луча",  "мощность 3-го луча",  "мощность 4-го луча", "мощность 5-го луча","мощность 6-го луча",  "мощность 7-го луча", "мощность 8-го луча", "мощность 9-го луча", "мощность 10-го луча", "мощность 11-го луча", "мощность 12-го луча", "мощность 13-го луча", "мощность 14-го луча", "мощность 15-го луча")]
    
    train_stats = train_data.describe()
    train_stats = train_stats.transpose()
    
    #нормализация входных данных
    def norm(x):
        return (x - train_stats['min']) /  (train_stats['max'] - train_stats['min'])
      
    model = LinearRegression()
    model.fit(norm(train_data), train_labels)
    
    from sklearn import metrics
    from sklearn.metrics import mean_squared_error
    from sklearn.metrics import mean_absolute_error

    y_test_for_x = np.array(test_labels.loc[:,'x'])
    
    # print("Прогнозы на тестовом наборе: {}".format(clf.predict(test_data)))
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
    x_error.extend(x_err)
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
plt.ylabel('y')
plt.xlabel('x')
plt.gca().invert_yaxis()
plt.show()