
import os
import csv
import logging as log
from datetime import datetime
from tensorflow.python.lib.io import file_io


class BaseAgent:
    def __init__(self, args):
        log.info('initializing agent...')
        self.args = args

    def write_results(self, fname, r_avg_list):
        fname = fname + '_' + datetime.now().strftime("%Y%m%d%H%M%S") + '.txt'
        fname = os.path.join(self.args.job_dir, fname)
        with file_io.FileIO(fname, mode='w+') as f:
            writer = csv.writer(f)
            writer.writerow(['timestamp', 'meanreward'])
            for r in r_avg_list:
                writer.writerow(r)
