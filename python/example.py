import os

from ioutils import process_tmp

if __name__ == "__main__":
    tmp = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tmp")
    process_tmp(tmp)
