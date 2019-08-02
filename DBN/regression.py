import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


np.random.seed(1337)  # for reproducibility 1337
from sklearn.datasets import load_boston
from sklearn.cross_validation import train_test_split
from sklearn.metrics.regression import r2_score, mean_squared_error
from sklearn.preprocessing import MinMaxScaler

from dbn.tensorflow import SupervisedDBNRegression

from openpyxl import load_workbook



# Loading dataset

peak_egat = pd.read_excel('', sheet_name='')

Y_raw = peak_egat[]
X_raw = peak_egat.iloc[:,:]


Y_test = Y_raw.iloc[]
Y_train = Y_raw.iloc[]

X_test = X_raw.iloc[]
X_train = X_raw.iloc[]



# Data scaling
min_max_scaler = MinMaxScaler()
X_train = min_max_scaler.fit_transform(X_train)

# Training
regressor = SupervisedDBNRegression(hidden_layers_structure=[7],
                                    learning_rate_rbm=0.01,
                                    learning_rate=0.0001,
                                    n_epochs_rbm=500,  
                                    n_iter_backprop=7409, # run more iter
                                    batch_size=40,
                                    activation_function='relu')
regressor.fit(X_train, Y_train)


# Test
X_test = min_max_scaler.transform(X_test)
Y_pred = regressor.predict(X_test)


mse = mean_squared_error(Y_test, Y_pred)
print('Done.\nR-squared: %f\nMSE: %f' % (r2_score(Y_test, Y_pred), mse))

def mean_absolute_percentage_error(y_test, y_pred): 
    y_test, y_pred = np.array(y_test), np.array(y_pred)
    #print('y_pred',y_pred)
    return np.mean(np.abs(y_test - y_pred)/y_test) * 100

Y_pred = np.array([y[0] for y in np.array(Y_pred)])
Y_test = np.array(Y_test)

mape = mean_absolute_percentage_error(Y_test, Y_pred)
print('Done. MAPE:', mape)



#write to excel

book = load_workbook('.xlsx')
writer = pd.ExcelWriter('.xlsx', engine='openpyxl') 
writer.book = book
writer.sheets = dict((ws.title, ws) for ws in book.worksheets)


Y_pred = pd.DataFrame(Y_pred)
Y_pred.to_excel(writer,"",header=False, index=False)

writer.save()








