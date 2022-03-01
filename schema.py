from typing import Optional
from pydantic import BaseModel


class APICALLBase(BaseModel):
    input: str
    output: str
    api_type: str
    CreateAt: str
    # geners: str
    # cast: str


class APICALLADD(APICALLBase):
    api_id: str

    class Config:
        orm_mode = True


class API(APICALLADD):
    id: int

    class Config:
        orm_mode = True
