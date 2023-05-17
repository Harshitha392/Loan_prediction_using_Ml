import requests
url = 'http://localhost:5000/predict_api'
r = requests.post(url,json={'gender':'Male', 'married':'NO', 'dependencies':0.0,'education':'Graduate','self_employed':'No','applicant_income':5849.0,'coapplicant_income':0.0,'loan_amount':128.0,'loan_amount_term':360.0,'credit_history':1.0,'property_area':'Urban'})
print(r.json())

#[['Male','No',0.0,'Graduate','No',5849.0,0.0,128.0,360.0,1.0,'Urban']]