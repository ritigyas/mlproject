import os
import sys
import dill
from src.exceptions import CustomException

def save_object(file_path: str, obj: object):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)   # <-- use file_obj, NOT file_path

    except Exception as e:
        raise CustomException(e, sys)