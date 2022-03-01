from pydantic import BaseModel

#### Class describes Bank Note
class BankNote(BaseModel):
    variance: float 
    skewness: float 
    curtosis: float 
    entropy: float
