from fastapi import FastAPI, Form
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import joblib
import pandas as pd




#app = object and fast api - constructor    we usse this because without initialization we 
# cant access fastapi methods


app=FastAPI()  #initialization of fast api class


model=joblib.load("mymodel.pkl")  #loading the model from the pickle file

# we write logic on fuction
#annotation-we giving instruction/rules(you have to work for this type of request and respnse)
#  to python function  @

@app.get("/")  #get method
def testing():
    return {"test":"all"}

@app.post("/prediction")
def myprediction(hours: float):

    newdata = pd.DataFrame({
        "StudyHours": [hours]
    })

    mynewdata = model.predict(newdata)
    print(mynewdata[0])

    if mynewdata[0] == 1:
        result = "Pass"
    else:
        result = "Fail"

    return {"prediction": result}