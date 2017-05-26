"""
DBCE - Diff Based Content Extraction

Copyright (c) 2016 Rafal Lesniak

This software is licensed as described in the file LICENSE.

"""

import logging
import json
import pdb
import os

from urllib.parse import urlparse
from collections import OrderedDict

from dbce.WarcIDX import WarcIDX
from dbce.WarcLoader import open_warc
from dbce.utils import get_layers

from bs4 import BeautifulSoup
from bs4 import Comment
import bs4.element

from datasketch import MinHash, MinHashLSH

logger = logging.getLogger(__name__)

def idx2dict(idx):
    """ Convert `pywb.cdx.cdxobject.CDXObject` to `dict` and extend
    it with informations which are needed later.

    Additional informations:
    - lengths are integers
    - urlpath
    - url_parsed - `urllib.parse.ParseResult`
    """

    _ = json.loads(idx.to_json())
    _['url_parsed'] = urlparse(idx['url'])
    _['urlpath'] = _['url_parsed'].path
    _['length'] = int(idx['length'])
    _['offset'] = int(idx['offset'])
    return _

def sanitize_bs4(soup):
    """Returns sanitized and prettified BeautifulSoup object as string

    This function removes all noscript, script, comments and iframs
    from the document.
    """
    bs4tag = bs4.element.Tag

    for _ in soup.findAll():
        if isinstance(_, bs4tag) and _.name in ['script', 'noscript', 'iframe']:
            _.extract()

    for _ in soup.findAll(text=lambda text: isinstance(text, Comment)):
        _.extract()

    if soup.body:
        for _ in getattr(soup, 'body', None).find_all('style'):
            _.extract()

    return soup.prettify()

def idx_to_journals(config, warc_input):
    journals = {}
    widx = WarcIDX(warc_input)
    # create journal
    for i in widx.index:
        url = i['url_parsed']
        if url.netloc not in journals:
            _ = journals.setdefault(url.netloc, DBCEJournal(config, domain=url.netloc))
        else:
            _ = journals[url.netloc]
        _.add_item(i)
    return journals

class DBCEJournal:
    """DBCEJournal - Journal holds als all informations needed for analysis and
    content extracion.

    Basically DBCEJournal is storage for faster and structured access to
    the documents we want to analyse.
    """

    def __init__(self, config, domain=None):
        assert domain
        self.domain = domain
        self.config = config
        self.warc_files = {}
        self._j_by_id = {}
        self.storage = {}

        self.item_count = 0
        self.lsh = None

    def get_by_id(self, jid):
        return self._j_by_id[jid]

    def compute_minhash(self, entry):
        """ Compute MinHash for given entry.

        The MinHash will be computed over `bs_html_sanitized` input starting from <body>
        and ending at </body>.
        """

        entry['minhash'] = MinHash(num_perm=self.config.getint(self.domain,
                                                               'NUMBER_OF_MINHASH_BUCKETS'))

        soup = entry['bs_html_sanitized']
        mhf_cnt = 0
        layers = get_layers(soup)
        # Stage 1 - compute MinHash values for all nodes in all layers
        for level in sorted(layers.keys()):
            node_group_nr = 0
            for nodes in layers[level]:
                node_nr = 0
                for node in nodes:
                    p_path = []
                    for _ in node.parentGenerator():
                        p_path.append("{}_{}_{}_{}".format(node_nr,
                                                           node_group_nr,
                                                           level,
                                                           _.name))

                    p_path.insert(0, "{}_{}_{}_{}".format(node_nr,
                                                          node_group_nr,
                                                          level,
                                                          node.name))

                    entry['minhash'].update('/'.join(p_path).encode('utf8'))
                    node_nr += 1

                mhf_cnt += 1
                node_group_nr += 1

        logger.info('MinHash computed based on {} features'.format(mhf_cnt))
        self.compute_lsh(entry)

    def compute_lsh(self, entry):
        """
        Indexes the WARC entry using LSH
        """

        if not self.lsh:
            self.lsh = MinHashLSH(
                threshold=self.config.getfloat(self.domain,
                                               'lsh_threshold'),
                num_perm=self.config.getint(self.domain,
                                            'number_of_minhash_buckets')
            )
        self.lsh.insert(entry['item_id'], entry['minhash'])

    def load_item(self, idx):
        """Load item idx from warcfile into the journal.

        Loaded html documents are parsed by BeautifulSoup, sanitized,
        MinHash and LSH computations will be done aswell.

        """

        idx_dict = idx2dict(idx)

        if idx_dict['url_parsed'].netloc != self.domain:
            return

        # compatibility to the Maschine
        idx_dict['warcfile'] = os.path.join(self.config.get(self.domain, 'warc_dir'),
                                            idx['filename'])

        idx_dict['item_id'] = self.item_count
        self.item_count += 1

        tmp = self.warc_files.setdefault(idx_dict['warcfile'],
                                         open_warc(idx_dict['warcfile']))
        logger.info('load entry start')

        idx_dict['warc_entry'] = tmp.load_entry(idx_dict['url'],
                                                offset=idx_dict['offset'],
                                                length=int(idx_dict['length']))

        logger.info('load entry end')
        idx_dict['raw_html'] = idx_dict['warc_entry'].data

        logger.info("parse html")

        idx_dict['bs_html'] = BeautifulSoup(idx_dict['raw_html'],
                                            self.config.get(self.domain, 'parser'))

        logger.info("sanitize html")
        idx_dict['bs_html_sanitized'] = BeautifulSoup(sanitize_bs4(idx_dict['bs_html']),
                                                      self.config.get(self.domain, 'parser'))

        logger.info("compute MinHash")
        self.compute_minhash(idx_dict)

        self._j_by_id[idx_dict['item_id']] = idx_dict
        return idx_dict

    def add_item(self, idx):
        """ Creates journal based on informations from index entry
        Internal storage for faster access of the items
        """

        # store the entry from warc file by path/ts/[...]
        idx_dict = self.load_item(idx)
        idx_dict['timestamp'] = int(idx_dict['timestamp'])

        _ = self.storage.setdefault(idx_dict['url_parsed'].path, OrderedDict())
        _ = _.setdefault(int(idx['timestamp']), idx_dict)

    def get_urlkeys(self):
        for _ in self.storage:
            yield _

    def get_paths(self):
        for urlkey in self.get_urlkeys():
            for _ in self.storage[urlkey]:
                yield (urlkey, _)

    def get_entry(self, path, ts):
        return self.storage[path][ts]

    def get_entry_by_id(self, eid):
        return self._j_by_id[eid]
