import datetime
from sqlmodel import SQLModel, Field, Session, select
from typing import Optional

from models.drill import DrillType
from models import engine

class UserSession(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    drill_type: Optional[DrillType] = Field(default=None)
    user_id: str
    start_time: datetime.datetime = Field(
        default_factory=lambda: datetime.datetime.now(datetime.timezone.utc)
    )
    end_time: Optional[datetime.datetime] = None
    active: bool = True


def create_user_session(user_id, first_name, last_name, drill_type: DrillType = None):
    user_session = UserSession(
        user_id=user_id, first_name=first_name, last_name=last_name, drill_type=drill_type
    )
    with Session(engine) as session:
        session.add(user_session)
        session.commit()

def end_user_session(user_id):
    with Session(engine) as session:
        statement = (
            select(UserSession)
            .where(UserSession.user_id == user_id)
            .where(UserSession.active == True)  # noqa: E712
        )
        user_session = session.exec(statement).one()
        if user_session:
            user_session.end_time = datetime.datetime.now(datetime.timezone.utc)
            user_session.active = False
            session.add(user_session)
            session.commit()
            session.refresh(user_session)

def get_all_active_user_sessions():
    with Session(engine) as session:
        statement = select(UserSession).where(UserSession.active == True)  # noqa: E712
        return session.exec(statement).all()

def update_drill_type(user_id, drill_type: DrillType):
    with Session(engine) as session:
        statement = select(UserSession).where(UserSession.user_id == user_id ).where(UserSession.active == True)  # noqa: E712
        user_session = session.exec(statement).one()
        if user_session:
            user_session.drill_type = drill_type
            session.add(user_session)
            session.commit()
            session.refresh(user_session)
            return user_session.dict()

def get_all_sessions_by_user_id(user_id):
    with Session(engine) as session:
        return session.exec(
            select(UserSession).where(UserSession.user_id == user_id)
        ).all()

def get_last_active_session_by_user_id(user_id):
    # Get recent Active session
    with Session(engine) as session:
        statement = (
            select(UserSession)
            .where(UserSession.user_id == user_id)
            .order_by(UserSession.start_time.desc())
        )
        data = session.exec(statement).first()
        print(data.dict())
        return data.dict()
