import numpy as np
from flask import Flask, request, jsonify, render_template
from copyreg import pickle
import pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sklearn
import pickle

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')



#####################################################################
def inversion1(data):
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
    matrix.iloc[:,11:14] = sc.fit_transform(matrix.iloc[:,11:14])
    return matrix
##############################################################################################################

@app.route('/predict',methods=['POST'])
def predict():
    
    a= request.form.get("gender")
    b=request.form.get("married")
    c=request.form.get("dependencies")
    d=request.form.get("education")
    e=request.form.get("self_employed")
    f=request.form.get("applicant_income")
    g=request.form.get("coapplicant_income")
    h=request.form.get("loan_amount")
    i=request.form.get("loan_amount_term")
    j=request.form.get("credit_history")
    k=request.form.get("property_area")
    list1=[]
    list1.extend([a,b,c,d,e,f,g,h,i,j,k])
    int_features1=inversion1([[a,b,c,d,e,f,g,h,i,j,k]])
    final_features = [np.array(int_features1)]
    prediction = model.predict(int_features1)
    output = prediction[0]
    print("output",output)
    if(output=='Y'):
        #return render_template('index.html', prediction_text='Loan is approved {}'.format(output))
        return render_template('index.html', prediction_text='Loan is approved')
    else:
        #return render_template('index.html', prediction_text='loan is not approved{}'.format(output))
        return render_template('index.html', prediction_text='Loan is not approved')
#####################################################
    """
    int_features1 = inversion1([x for x in request.form.values()])
    # list1=([x for x in request.form.values()])
    list1=[]
    #int_features1 = inversion1(list1)
    final_features = [np.array(int_features1)]
    for i in list1:
        print(i)
    prediction = model.predict(int_features1)
    """
    # prediction = model.predict(final_features)
    """
    output = prediction[0]
    if(output=='Y'):
        #return render_template('index.html', prediction_text='Loan is approved {}'.format(output))
        return render_template('index.html', prediction_text='Loan is approved')
    else:
        #return render_template('index.html', prediction_text='loan is not approved{}'.format(output))
        return render_template('index.html', prediction_text='Loan is not approved')
        """
####################################################################################
    # return render_template('index.html', prediction_text='Loan is approved')

@app.route('/predict_api',methods=['POST'])
def predict_api():
    data = request.get_json(force=True)
    prediction = model.predict([np.array(list(data.values()))])
    output = prediction[0]
    return jsonify(output)

if __name__ == "__main__":
    app.run(debug=True)