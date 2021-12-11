# WIP: LOAD TABLES TO POSTGRES
import pandas as pd
from sqlalchemy import create_engine

db_name = 'ccer'
user = 'jose'
password = ''

# WIP 
engine = create_engine(f'postgresql://{user}:{password}@localhost/{db_name}', echo=False) # Find a way to use this in messages

psql_conn = engine.connect()
