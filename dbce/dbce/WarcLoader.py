"""
WARC realted functions and objects.

Copyright (c) 2016 Rafal Lesniak

This software is licensed as described in the file LICENSE.

"""

import logging
import cgi
import pywb.warc.archiveiterator

BUFF_SIZE = 4096

class WorkObject:
    def __init__(self, uri, data, url=None, charset='utf-8'):
        self.uri = uri
        self.url = url
        self.wlid = uri.replace('<', '').replace('>', '').split(':', 2)[2]
        self.data = data
        self.charset = charset

class WarcLoader:

    def __init__(self, warc=None, logger=None):
        self.warcfile = warc
        self.logger = (logger or logging.getLogger(__name__))
        self.logger.info("warcfile: {}".format(warc))
        self.entry = None
        if warc:
            infile = open(self.warcfile, 'rb')
            self.architer = pywb.warc.archiveiterator.ArchiveIterator(infile)
        else:
            self.architer = None

    def load_entry(self, url, offset, length):
        self.logger.info("Load entry ({}, {}, {}) from {}".format(url, offset, length, self.warcfile))
        entry = None

        def stream2buf(stream):
            while True:
                buff = stream.read(BUFF_SIZE)
                yield buff
                if not buff:
                    break

        self.architer.fh.seek(offset)

        for i in self.architer():
            if self.architer.offset != offset:
                continue

            if i.rec_type in ['conversion']:
                raise TypeError("Entry is {}".format(i.rec_type))

            if i.rec_type not in ['response', 'resource']:
                continue

            header_params = cgi.parse_header(i.status_headers.get_header('Content-Type'))

            if 'charset' in header_params:
                _charset = header_params['charset']
            else:
                _charset = 'utf-8'

            entry = WorkObject(i.rec_headers.get_header('WARC-Record-ID'), b'', url=url,
                               charset=_charset)

            for _ in stream2buf(i.stream):
                entry.data += _

            break

        if not entry:
            raise ValueError("Entry not found")

        return entry

def open_warc(warcfile=None):
    """Returns WarcLoader for given WARC file"""
    return WarcLoader(warcfile)
