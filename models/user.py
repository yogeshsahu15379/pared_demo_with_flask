from typing import Optional

import datetime
import uuid
from sqlmodel import SQLModel, Field, Session, delete, select

from models.drill import DrillType
from config import engine


class User(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    user_id: str = Field(..., unique=True)
    first_name: str
    last_name: str
    created_at: datetime.datetime = Field(
        default_factory=lambda: datetime.datetime.now(datetime.timezone.utc)
    )
    updated_at: datetime.datetime = Field(
        default_factory=lambda: datetime.datetime.now(datetime.timezone.utc)
    )

def create_user(user_id, first_name, last_name):
    user = User(user_id=user_id, first_name=first_name, last_name=last_name)
    with Session(engine) as session:
        session.add(user)
        session.commit()

def get_all_users():
    with Session(engine) as db:
        return db.exec(select(User)).all()
    
def get_user_by_id(user_id):
    with Session(engine) as db:
        return db.exec(select(User).where(User.user_id == str(user_id))).first()