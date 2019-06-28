import os
import gzip
def get_file_extension(infile):
    filename = os.path.split(infile)[1]
    filename_arr = filename.split(".")
    return filename_arr[len(filename_arr)-1]

def open_file(filename):
    f = open(filename, 'rb')
    is_gzipped = f.read(2) == b'\x1f\x8b'
    f.close()
    r = gzip.open(filename, 'rt') if is_gzipped else open(filename, 'rt')
    return r
