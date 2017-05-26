"""
CrossLSH - Datatypes related to the cross

Copyright (c) 2016 Rafal Lesniak

This software is licensed as described in the file LICENSE.
"""

class CrossLSH:
    """
    CrossLSH - container representing the datastructure
    used for analysis in DBCE-Method
    """

    def __init__(self):
        self.cur = None
        self.simup = None
        self.simdown = None
        self.next = None
        self.prev = None
        self.window_size = 0

    def to_bits(self):
        """Return cross configuration as bit array"""
        bits = [0] * 5
        if self.cur:
            bits[0] = 1
        if self.simup:
            bits[1] = 1
        if self.simdown:
            bits[2] = 1
        if self.prev:
            bits[3] = 1
        if self.next:
            bits[4] = 1
        return tuple(bits)

    def to_csv(self):
        header = ['cross_cur_docid',
                  'cross_cur_ts',
                  'cross_simup_docid',
                  'cross_simup_ts',
                  'cross_simdown_docid',
                  'cross_simdown_ts',
                  'cross_next_docid',
                  'cross_next_ts',
                  'cross_prev_docid',
                  'cross_prev_ts',
                  'cross_window_size',
                  'cross_bits',
                  'cross_unique_id']

        return (header, [self.cur['warc_entry'].wlid,
                         self.cur['timestamp'],
                         self.simup['warc_entry'].wlid if self.simup else None,
                         self.simup['timestamp'] if self.simup else None,
                         self.simdown['warc_entry'].wlid if self.simdown else None,
                         self.simdown['timestamp'] if self.simdown else None,
                         self.prev['warc_entry'].wlid if self.prev else None,
                         self.prev['timestamp'] if self.prev else None,
                         self.next['warc_entry'].wlid if self.next else None,
                         self.next['timestamp'] if self.next else None,
                         self.window_size,
                         "".join(map(str, self.to_bits())),
                         self.get_unique_id()])

    def get_unique_id(self):
        """ Retrun the unique ID of the cross based on
        the documents available."""

        _tmp_id = []
        for (bit, crossel) in zip(self.to_bits(), [self.cur, self.simup,
                                                   self.simdown, self.prev,
                                                   self.next]):
            # we take warc entry id
            if bit:
                _tmp_id.append(abs(hash(crossel['warc_entry'].wlid)))
            # but if it's not present we take random token
            else:
                _tmp_id.append(abs(hash('bew1z4ulel')))

        return "-".join(map(str, _tmp_id))
