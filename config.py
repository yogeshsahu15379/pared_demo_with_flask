import os

from sqlmodel import create_engine

class Config:
    # Database configuration
    DB_FILE_NAME = os.environ.get("DB_NAME", "test_db.db")
    # Flask configuration
    SECRET_KEY = os.environ.get("SECRET_KEY", "some_random_generated_key")
    DEBUG = os.environ.get("FLASK_DEBUG", "False") == "True"

    # PATHS
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

engine = create_engine(f"sqlite:///{Config.DB_FILE_NAME}", echo=False)
