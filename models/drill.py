import datetime
from enum import Enum
from typing import Optional

from sqlmodel import SQLModel, Field

class DrillType(str, Enum):
    SALUTE = "SALUTE"
    TEJ_CHAL = "TEJ_CHAL"
    BAJU_SWING = "BAJU_SWING"
    KADAMTAL = "KADAMTAL"
    SLOW_CHAL = "SLOW_CHAL"
    HILL_MARCH = "HILL_MARCH"

class BaseDrill(SQLModel):
    id: Optional[int] = Field(default=None, primary_key=True)
    timestamp: datetime.datetime = Field(
        default_factory=lambda: datetime.datetime.now(datetime.timezone.utc)
    )
    angle: float
    palm_angle: float
    status: Optional[str] = None
    suggestion: Optional[str] = None
    screenshot_path: Optional[str] = None
    drill_type: DrillType
    user_id: str
    session_id: str
