import uvicorn
import pickle
from fastapi import  Depends, FastAPI, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session
from typing import List

# import model as classifier
# import BankNote
import schema
import crud
from db_connection import SessionLocal, engine

# Class describes Bank Note
class BankNote(BaseModel):
    variance: float 
    skewness: float 
    curtosis: float 
    entropy: float


# Create the app
app = FastAPI(
    title="Bank Note Prediction",
    description="You can predict bank note is fake or true by a Machine Learnig Model",
    version="1.0.0"
)

# Dependency
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

pickle_in = open("model_bin","rb")
classifier=pickle.load(pickle_in)


# Home route
@app.get('/')
def index():
    return {'message': 'Hello, World'}


@app.get('/all_data', response_model=List[schema.Table])
def retrieve_all_data(skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    data = crud.get_data(db=db, skip=skip, limit=limit)
    return data

@app.post('/add_data', response_model=schema.DataAdd)
def add_new_data(api: schema.DataAdd, db: Session = Depends(get_db)):
    # api_id = crud.get_data_by_api_id(db=db, api_id=api.api_id)
    # if api_id:
    #     raise HTTPException(status_code=400, detail=f"Movie id {api.movie_id} already exist in database: {api_id}")
    return crud.add_data_details_to_db(db=db, api=api)


# Prediction Route
@app.post('/prediction')
def predict_banknote(data:BankNote):
    data = data.dict()
    variance=data['variance']
    skewness=data['skewness']
    curtosis=data['curtosis']
    entropy=data['entropy']
    prediction = classifier.predict([[variance,skewness,curtosis,entropy]])
    if(prediction[0]>0.5):
        prediction="Fake note"
    else:
        prediction="Its a Bank note"
    return {
        'prediction': prediction
    }

    
if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=8000)