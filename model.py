from copyreg import pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sklearn
import pickle
#importing dataset
dataset = pd.read_csv('Customer_data.csv', header = 0)
x = dataset.iloc[:,1:-1].values
y = dataset.iloc[:,-1].values

#Data Pre Processing Function
def DataPreProcessing(data):
    from sklearn.impute import SimpleImputer
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.preprocessing import LabelEncoder

    imputer = SimpleImputer(missing_values = np.nan,strategy = 'most_frequent')
    imputer2 = SimpleImputer(missing_values = np.nan, strategy = 'mean')

    imputer.fit(x[:,[0,1,2,4,9]])
    data[:,[0,1,2,4,9]] = imputer.transform(data[:,[0,1,2,4,9]])

    imputer2.fit(data[:,7:9])
    data[:,7:9] = imputer2.transform(data[:,7:9])

    ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [2,10])], remainder='passthrough')
    data = np.array(ct.fit_transform(data))
    data = pd.DataFrame(data)

    data[7] = LabelEncoder().fit_transform(data[7])
    data[8] = LabelEncoder().fit_transform(data[8])
    data[9] = LabelEncoder().fit_transform(data[9])
    data[10] = LabelEncoder().fit_transform(data[10])
    return data

#Function Call
x = DataPreProcessing(x)

#Splitting the Dataset
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.20, random_state=42)

#Standardization
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train.iloc[:, 11:14] = sc.fit_transform(x_train.iloc[:, 11:14])
x_test.iloc[:, 11:14] = sc.transform(x_test.iloc[:, 11:14])

#Machine Learn Model Using Logistic Regression
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0, max_iter = 500)
classifier.fit(x_train, y_train)

pickle.dump(classifier, open("model.pkl","wb"))

#Predictions
y_pred = classifier.predict(x_test)

#Accuracy of the model
from sklearn.metrics import confusion_matrix, accuracy_score
print(accuracy_score(y_test,y_pred))

#Function to invert Data for the model
def Inversion(data):
    if(data[0][0] == 'Male'):
        data[0][0] = 1
    else:
        data[0][0] = 0

    if(data[0][1] == 'No'):
        data[0][1] = 0
    else:
        data[0][1] = 1

    if(data[0][3] == 'Graduate'):
        data[0][3] = 0
    else:
        data[0][3] = 1

    if(data[0][4] == 'No'):
        data[0][4] = 0
    else:
        data[0][4] = 1
    
    area = []
    dependents = []
    if(data[0][10] == 'Rural'):
        area = [1,0,0]
    elif(data[0][10] == "Semiurban"):
        area = [0,1,0]
    else:
        area = [0,0,1]
    

    if(data[0][2] == 0):
        dependents = [1,0,0,0]
    elif(data[0][2] == 1):
        dependents = [0,1,0,0]
    elif(data[0][2] == 2):
        dependents = [0,0,1,0]
    else:
        dependents = [0,0,0,1]
    rem = []
    for i in range(11):
        if(i != 10 and i != 2):
            rem.append(data[0][i])
        
    final = dependents + area + rem
    matrix = []
    matrix.append(final)
    matrix = pd.DataFrame(matrix)
    matrix.iloc[:,11:14] = sc.transform(matrix.iloc[:,11:14])
    return matrix


# pickle.dump(classifier,open('model.pkl','wb'))
# model=pickle.loan(open('model.pkl'),'rb')
# print(model.predict([['Male','No',0.0,'Graduate','No',5849.0,0.0,128.0,360.0,1.0,'Urban']]))


#Custom Predictions

# new_x = [['Male','No',0.0,'Graduate','No',5849.0,0.0,128.0,360.0,1.0,'Urban']]
# single_pred = Inversion(new_x)
# ans = classifier.predict(single_pred)
# if(ans == 'Y'):
#     print("YES")
# else:
#     print("NO")

#