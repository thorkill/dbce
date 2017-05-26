
import gzip
from urllib.parse import urlparse
from pywb.cdx.cdxobject import CDXObject

def open_file(filename):
    """ Returns filehandle

    Shortcut for accessing files:
    if filename ends with .gz use gzip
    else use open()"""

    if filename.endswith('.gz'):
        fd = gzip.GzipFile(filename)
    else:
        fd = open(filename, 'rb')
    return fd

class WarcIDX:

    """ WarcIDX class for accessing the WARC index files """

    def __init__(self, idx_sources=None):
        self.sources = []

        if isinstance(idx_sources, (str, bytes)):
            self.sources.append(idx_sources)

        elif isinstance(idx_sources, (list, tuple)):
            for i in idx_sources:
                self.sources.append(i)
        else:
            raise ValueError("idx_sources type {} not supported".format(type(idx_sources)))

    @property
    def index(self):
        for filename in self.sources:
            with open_file(filename) as fd:
                for line in fd:
                    idx = CDXObject(line)
                    idx['url_parsed'] = urlparse(idx['url'])
                    yield idx
                fd.close()
