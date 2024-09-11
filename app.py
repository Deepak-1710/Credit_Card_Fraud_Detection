from flask import Flask, render_template, request

app = Flask(__name__)# interface between my server and my application wsgi

import pickle
model = pickle.load(open(r'C:\Users\annab\Downloads\ADS_PROJECT\model.pkl','rb'))

@app.route('/')#binds to an url
def helloworld():
    return render_template("index.html")

@app.route('/login', methods =['POST'])#binds to an url
def login():
    p =request.form['Amount']
    v1=request.form['V1']
    v2=request.form['V2']
    v3=request.form['V3']
    v4=request.form['V4']
    v5=request.form['V5']
    v6=request.form['V6']
    v7=request.form['V7']
    v8=request.form['V8']
    v9=request.form['V9']
    v10=request.form['V10']
    v11=request.form['V11']
    v12=request.form['V12']
    v13=request.form['V13']
    v14=request.form['V14']
    v15=request.form['V15']
    v16=request.form['V16']
    v17=request.form['V17']
    v18=request.form['V18']
    v19=request.form['V19']
    v20=request.form['V20']
    v21=request.form['V21']
    v22=request.form['V22']
    v23=request.form['V23']
    v24=request.form['V24']
    v25=request.form['V25']
    v26=request.form['V26']
    v27=request.form['V27']
    v28=request.form['V28']
    t=[[float(v1),float(v2),float(v3),float(v4),float(v5),float(v6),float(v7),float(v8),float(v9),float(v10),float(v11),float(v12),float(v13),float(v14),float(v15),float(v16),float(v17),float(v18),float(v19),float(v20),float(v21),float(v22),float(v23),float(v24),float(v25),float(v26),float(v27),float(v28),float(p)]]
    output= model.predict(t)
    if output==1:
        data="Fraud"
    else:
        data="Not fraud"
    
    return render_template("index.html",y = "=" + data )
    


if __name__ == '__main__' :
    app.run(debug= False)# -*- coding: utf-8 -*-

