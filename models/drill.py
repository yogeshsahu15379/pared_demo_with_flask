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

    def __str__(self):
        return self.value

# TEMPORARY
# Forward map: slug -> DrillType
DRILL_SLUG_MAP = {
    "salute": DrillType.SALUTE,
    "tejchal": DrillType.TEJ_CHAL,
    "bajuswing": DrillType.BAJU_SWING,
    "kadamchal": DrillType.KADAMTAL,
    "slowmarch": DrillType.SLOW_CHAL,
    "hillmarch": DrillType.HILL_MARCH,
}

# Reverse map: DrillType -> slug
DRILL_TYPE_SLUG_MAP = {v: k for k, v in DRILL_SLUG_MAP.items()}

# New mapping: DrillType -> Script file
DRILL_TYPE_SCRIPT_MAP = {
    DrillType.SALUTE: "salute_detection.py",
    DrillType.KADAMTAL: "kadamchal_detection.py",
    DrillType.BAJU_SWING: "baju_swing_detection.py",
    DrillType.TEJ_CHAL: "tej_chal_detection.py",
    DrillType.SLOW_CHAL: "slow_march_detection.py",
    DrillType.HILL_MARCH: "hill_march_detection.py",
}

DRILL_CAMERA_URL_MAP = {
    DrillType.SALUTE: "rtsp://admin:admin@123@192.168.0.14:554/1/2?transmode=unicast&profile=vam",
    # DrillType.SALUTE: "rtsp://192.168.1.100:8080/h264_ulaw.sdp?transmode=unicast&profile=vam",
    DrillType.KADAMTAL: "XXXXXXXXXXXXXXXXXXXXXXXXXXXXX",
    DrillType.BAJU_SWING: "XXXXXXXXXXXXXXXXXXXXXXXXXXXXX",
    DrillType.TEJ_CHAL: "XXXXXXXXXXXXXXXXXXXXXXXXXXXXX",
    DrillType.SLOW_CHAL: "XXXXXXXXXXXXXXXXXXXXXXXXXXXXX",
    DrillType.HILL_MARCH: "XXXXXXXXXXXXXXXXXXXXXXXXXXXXX",
}

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
