import numpy as np
import pandas as pd
from pandas import Series, DataFrame
import sqlalchemy as db

from data.fish_api import getFishData

engine = db.create_engine("mariadb+mariadbconnector://python:python1234@127.0.0.1:3306/pythondb")

new_fishs = getFishData()
# print(new_fishs)
train_fishs = new_fishs[:35]
test_fishs = new_fishs[35:]

train_fishs_dataFrame = pd.DataFrame(train_fishs, columns=["train_lenght", "train_weight", "train-target"])
test_fishs_dataFrame = pd.DataFrame(test_fishs, columns=["test_lenght", "test_weight", "test-target"])
print(train_fishs_dataFrame)
print(test_fishs_dataFrame)

def insert():
    train_fishs_dataFrame.to_sql("train", engine, index=False, if_exists="replace")
    test_fishs_dataFrame.to_sql("test", engine, index=False, if_exists="replace")

insert()
