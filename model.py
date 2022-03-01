from sqlalchemy import Boolean, Column, Integer, String
from db_handler import Base


class API(Base):
    __tablename__ = "api_data"
    id = Column(Integer, primary_key=True, autoincrement=True, index=True, nullable=False)
    api_id = Column(String, index=True, nullable=False)
    input = Column(String, index=True, nullable=False)
    output = Column(String(100), index=True, nullable=False)
    api_type = Column(String(100), index=True, nullable=False)
    CreateAt = Column(String(100), index=True, nullable=False)
