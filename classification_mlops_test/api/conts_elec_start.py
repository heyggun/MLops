import os
os.system(
    """/home/yeoai/anaconda3/envs/pn_classification/bin/gunicorn src.run:app --config ./src/utils/config/gconf.py"""
)
