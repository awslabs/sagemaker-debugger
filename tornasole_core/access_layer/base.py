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


class TSAccessFile(TSAccessBase):
    def __init__(self, path, mode):
        self.path = path
        self.mode = mode
        self.open(path,mode)

    def open(self,path,mode):
        self._accessor = open(path, mode)

    def write(self, _str):
        print( "writing", len(_str))
        self._accessor.write(_str)

    def flush(self):
        self._accessor.flush()

    def close(self):
        self._accessor.close()

    def ingest_all(self):
        self._data = self._accessor.read()
        self._datalen = len(self._data)
        self._position = 0
        print( "Ingesting All", self._datalen)
    
    def read(self,n):
        print( f'Pos={self._position}, N={n}, DataLen={self._datalen}')
        assert self._position+n <= self._datalen
        res = self._data[self._position:self._position+n]
        self._position += n
        return res

    def has_data(self):
        return self._position < self._datalen

