from cmath import log
from typing import List

from fastapi import Depends, FastAPI, HTTPException
from sqlalchemy.orm import Session
from fastapi.middleware.cors import CORSMiddleware

import uvicorn
import pickle
import crud
import model
import schema
from db_handler import SessionLocal, engine
from pydantic import BaseModel
from datetime import datetime

now = datetime.now()

model.Base.metadata.create_all(bind=engine)


# Class describes Bank Note
class BankNote(BaseModel):
    variance: float 
    skewness: float 
    curtosis: float 
    entropy: float


# initiating app
app = FastAPI(
    title="Bank Prediction App",
    description="You can perform CRUD operation by using this API",
    version="1.0.0"
)

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

pickle_in = open("model_bin","rb")
classifier=pickle.load(pickle_in)


# Dependency
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@app.get('/', response_model=List[schema.API])
def retrieve_all_api_called(skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    api = crud.get_all_api_called(db=db, skip=skip, limit=limit)
    print('=================', api)
    return api


# Prediction Route
@app.post('/prediction')
def predict_banknote(data:BankNote, db: Session = Depends(get_db)):
    print('data typeeee  -----', type(data))
    data = data.dict()
    variance=data['variance']
    skewness=data['skewness']
    curtosis=data['curtosis']
    entropy=data['entropy']
    prediction = classifier.predict([[variance,skewness,curtosis,entropy]])

    print('prediction typee ----', type(prediction))
    if(prediction[0]>0.5):
        prediction="Fake note"
    else:
        prediction="Its a Bank note"

    crud.add_data_into_db(db=db, api_id='1', api=data, prediction=prediction, api_type='Prediction of bank note', time=f'{now}')

    return {
        'prediction': prediction
    }


if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=3030)