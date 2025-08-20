from sqlmodel import SQLModel, create_engine

# This is your SQLite .db file
# sqlite_file_name = "salute_results.db"
sqlite_file_name = "sample_data.db"
sqlite_url = f"sqlite:///{sqlite_file_name}"

# Create the engine
engine = create_engine(sqlite_url, echo=True)  # echo=True to see generated SQL logs

def init_db():
    SQLModel.metadata.create_all(engine)
