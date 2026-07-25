import sys

from logger import logging
from exceptions import CustomException

if __name__ == "__main__":
    try:
        logging.info("Starting the test")

        a = 10
        b = 0
        c = a / b   # This will raise ZeroDivisionError

    except Exception as e:
        logging.error(CustomException(e, sys))