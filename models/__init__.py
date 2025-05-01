from sqlmodel import SQLModel, create_engine

def init_db(app):
    # This is your SQLite .db file
    sqlite_url = f"sqlite:///{app.config["DB_FILE_NAME"]}"

    # Create the engine
    engine = create_engine(sqlite_url, echo=True)
    SQLModel.metadata.create_all(engine)

