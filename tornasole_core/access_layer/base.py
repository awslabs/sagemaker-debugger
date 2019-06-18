import os

class TSAccessBase:
    def __init__(self):
        pass

    def open(self):
        raise NotImplementedError

    def write(self,_bytes):
        raise NotImplementedError

    def flush(self):
        raise NotImplementedError

    def close(self):
        raise NotImplementedError

    def has_data(self):
        raise NotImplementedError        

