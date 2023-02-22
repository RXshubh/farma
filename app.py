from flask import Flask,request,jsonify
import pickle
import numpy as np

model=pickle.load(open('RandomForestAPK.pkl','rb'))

app=Flask(__name__)

@app.route('/predict',methods=['POST'])
def predict():
    Temp=request.form.get('Temp')
    Humidity=request.form.get('Humidity')
    Rainfall=request.form.get('Rainfall')
    pH=request.form.get('pH')
    N = request.form.get('N')
    P=request.form.get('P')
    K = request.form.get('K')
    Ca = request.form.get('Ca')
    Mg = request.form.get('Mg')
    S = request.form.get('S')
    Fe = request.form.get('Fe')
    Mn = request.form.get('Mn')
    Zn = request.form.get('Zn')
    Cu = request.form.get('Cu')
    Na = request.form.get('Na')




    

    for_input = np.array([[Temp, Humidity, Rainfall, pH,N,P,K,Ca,Mg,S,Fe,Mn,Zn,Cu,Na]])
    result = model.predict(for_input)[0]


    return jsonify({'Reccomendation': str(result)})


if __name__=='__main__':
    app.run(debug=True)
