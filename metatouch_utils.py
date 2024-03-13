#!/usr/bin/env python3

import sys
import os
import configparser

def purge_data():
"""Deletes any files that have .npy, .csv, .png extensions"""
    dir = os.getcwd()
    target = [".npy",".csv",".png"]
    for item in os.listdir(dir):
        for file_type in target:
            if item.endswith(file_type):
                os.remove(os.path.join(dir, item))
                break


if __name__ == __main__:

