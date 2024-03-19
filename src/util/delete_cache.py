# cmd application that deletes files in cache folder that are younger than 1 day

import os
import time
import argparse

from src.util.constants import Directory

def delete_cache(seconds: int, folder: str):

    for file in (Directory.CACHING_DIR / folder).rglob('*'):

        if os.stat(file).st_mtime > time.time() - seconds:

            if os.path.isfile(file):
                print(file)
                file.unlink()

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Delete files in cache folder that are younger than a specified time')
    parser.add_argument('--seconds', type=int, default=86400, help='Seconds')
    parser.add_argument('--folder', type=str, default='cache', help='Folder')

    args = parser.parse_args()

    delete_cache(
        seconds=args.seconds,
        folder=args.folder
    )