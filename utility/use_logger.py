# -*- coding: utf-8 -*-
# ------------------------------------
# Create On 2018/9/19
# File Name: use_logger
# Edit Author: lnest
# ------------------------------------
import os
import logging
# import conf.settings as config
# from logging.handlers import TimedRotatingFileHandler

PID = os.getpid()
CURRENT_DIRECTORY = os.path.dirname(os.path.abspath(__file__))  # get absolutely directory of this file


def set_log_level(level=logging.DEBUG, logger_name=''):
    # console output
    logger = logging.getLogger(logger_name)
    logger.setLevel(level)
    console = logging.StreamHandler()
    console.setLevel(level)
    formatter = logging.Formatter('%(asctime)s: %(levelname)-8s: %(filename)-15s:%(lineno)3d\t%(message)s')
    console.setFormatter(formatter)
    logger.addHandler(console)


# def get_logger(logger_name=config.LOGGER_NAME):
#     logger = logging.getLogger(logger_name)
#     return logger