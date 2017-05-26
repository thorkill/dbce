
"""
Copyright (c) 2016 Rafal Lesniak

This software is licensed as described in the file LICENSE.
"""

from collections import OrderedDict

class NodeStats(OrderedDict):
    pass

class NodeInfo(object):

    def __init__(self, node):
        self.node = None
        self.dbce_class = None
        self.dummy_node = None
        self.node_name = None
        self.nodeid = None
        self.parent = None
        self.preprocess_class = None
        self.xpath = None
        self.xpath_hash = None
        self.clf_vector = None
        self.stats = None
        self.stats_done = None

        self.link_density = None
        self.text_density = None

        self._init(node)

    def _init(self, node):
        self.add_html_info = {}
        self.node = node
        self.dummy_node = None
        self.node_name = self.node.name
        self.nodeid = id(self.node)
        self.parent = self.node.parent
        self.preprocess_class = None
        self._xpath = None
        self._xpath_hash = None
        self.clf_vector = None
        self.stats = {
            'href_cnt': 0,
            'num_wraped_lines': 0,
            'tag_cnt': 0,
            'text_in_links': 0,
            'text_len': 0,
            'word_cnt': 0,
        }
        self.stats_done = False

        self.link_density = -1
        self.text_density = -1

    def sum_stats(self, other):
        for _ in self.stats:
            self.stats[_] += other.stats[_]

    # FIXME: we have header redundant in here - make get_csv_header() to_csv()
    def to_csv(self, delimiter=','):
        columns = ('nodeid', 'node_name', 'preprocess_class',
                   'dbce_class', 'text_density', 'link_density', 'xpath', 'xpath_hash')

        columns_node_stats = ('href_cnt', 'num_wraped_lines', 'tag_cnt', 'text_in_links',
                              'text_len', 'word_cnt')

        _ = [str(getattr(self, x, '')) for x in (
            'nodeid', 'node_name', 'preprocess_class', 'dbce_class',
            'text_density', 'link_density', 'xpath', 'xpath_hash')] \
            + [str(self.stats.get(x, '')) for x in columns_node_stats]
        return (columns + columns_node_stats, delimiter.join(_))
