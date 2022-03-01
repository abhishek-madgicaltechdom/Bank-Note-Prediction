from sqlalchemy.orm import Session
import model
import schema
import json


def get_all_api_called(db: Session, skip, limit):
    return db.query(model.API).offset(skip).limit(limit).all()


def add_data_into_db(db: Session, api_id, api: schema.APICALLADD, prediction, api_type, time):
    print('movie---', api, prediction, api_type, time)
    data = model.API(
        api_id=api_id,
        input=json.dumps(api),
        output=prediction,
        api_type=api_type,
        CreateAt=time
        
    )
    db.add(data)
    db.commit()
    db.refresh(data)
