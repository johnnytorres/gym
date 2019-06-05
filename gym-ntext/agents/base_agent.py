
import os
import csv
import logging as log
from datetime import datetime
from tensorflow.python.lib.io import file_io


class BaseAgent:
    def __init__(self, args):
        log.info('initializing agent...')
        dateTimeObj = datetime.now()
        self.file_id = dateTimeObj.strftime("%Y%m%d%H%M%S")
        self.args = args

    def write_results(self, fname, r_avg_list):
        fname = fname + '_' + self.file_id + '.txt'
        fname = os.path.join(self.args.job_dir, fname)
        write_header = not os.path.exists(fname)
        with file_io.FileIO(fname, mode='w+') as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow(['run_id', 'timestamp', 'epoch', 'metric', 'value'])
            for r in r_avg_list:
                writer.writerow(r)
