import datetime
from sqlmodel import SQLModel, Field, Session, select
from typing import Optional

from models.drill import DrillType, DRILL_SLUG_MAP
from config import engine
from models.user import get_user_by_id

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


def create_user_session(user_id, drill_type: DrillType = None):
    user_session = UserSession(
        user_id=user_id, drill_type=drill_type
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
    sessions_with_name = []
    with Session(engine) as session:
        statement = select(UserSession).where(UserSession.active == True)  # noqa: E712
        sessions = session.exec(statement).all()

        for session in sessions:
            user = get_user_by_id(session.user_id)
            if user:
                session.first_name = user.first_name
                session.last_name = user.last_name
                sessions_with_name.append(session)
        return sessions_with_name

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
        if data:
            return data.dict()

def get_all_user_sessions(drill_type: DrillType | None = None) -> list[dict]:
    with Session(engine) as db:
        stmt = select(
            UserSession.user_id, UserSession.first_name, UserSession.last_name
        )
        if drill_type:
            stmt = stmt.where(UserSession.drill_type == drill_type)
        stmt = stmt.distinct(UserSession.user_id)
        raw = db.exec(stmt).all()
        return [{"user_id": u[0], "first_name": u[1], "last_name": u[2]} for u in raw]
    
def get_sessions_for_user(
    user_id: str, drill_type: DrillType | None = None
) -> list[UserSession]:
    with Session(engine) as db:
        stmt = select(UserSession).where(UserSession.user_id == user_id)
        if drill_type:
            stmt = stmt.where(UserSession.drill_type == drill_type)
        stmt = stmt.order_by(UserSession.start_time.desc())
        return db.exec(stmt).all()
    
def get_session_by_id(session_id: int) -> UserSession:
    with Session(engine) as db:
        return db.get(UserSession, session_id)