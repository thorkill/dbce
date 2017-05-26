#!/usr/bin/env python3

"""
Copyright (c) 2016 Rafal Lesniak

This software is licensed as described in the file LICENSE.
"""

import logging.config
import pdb
import unittest
import os
import bs4.element
import random
from urllib.parse import urlparse
from bs4 import BeautifulSoup

from dbce.DBCEJournal import (DBCEJournal, idx_to_journals)
from dbce.DBCEMethod import (DBCEMethodLSH, stage2_step_learn,
                              wrap_with_div,
                              percentage, get_layers, extract_content)
from dbce.experiment import MaschineLSH
from dbce.WarcIDX import WarcIDX
from dbce.constants import (DBCE_CLASS_CONTENT, DBCE_CLASS_UNKNWON,
                            DBCE_CLASS_BOILER_PLATE, DBCE_CLASS_INVALID,
                            DBCE_DIFF_CUR, DBCE_DIFF_UP)
from dbce.classification import classify_block
from dbce.utils import (ratio, filter_tag, get_working_dir, dump_nodes_info, create_working_dir)

from dbce.datatypes.context import DBCECTX
from dbce.datatypes.cross import CrossLSH

logging.config.fileConfig('../logging.conf')

import configparser
config = configparser.ConfigParser()
config.read('../dbce.ini')

SLOW_TESTS = os.environ.get('SLOW_TESTS', False)

# FIXME: all asserts should be
# assertEqual(shouldbe, is)

class TestClassification(unittest.TestCase):

    def test_classify_cur_up_00000(self):
        versions = (1, 1, 0, 0, 0)
        # element not found in both versions
        bits = (0, 0, 0, 0, 0)
        with self.assertRaises(ValueError):
            self.assertEqual(DBCE_CLASS_INVALID, classify_block(versions, bits))

    def test_classify_cur_up_10000(self):
        versions = (1, 1, 0, 0, 0)
        # element is only in `cur`
        bits = (1, 0, 0, 0, 0)
        self.assertEqual(DBCE_CLASS_CONTENT, classify_block(versions, bits))

    def test_classify_cur_up_11000(self):
        versions = (1, 1, 0, 0, 0)
        # element is in `cur` and `up`
        bits = (1, 1, 0, 0, 0)
        self.assertEqual(DBCE_CLASS_BOILER_PLATE, classify_block(versions, bits))

    def test_classify_cur_up_down_10000(self):
        versions = (1, 1, 1, 0, 0)
        bits = (1, 0, 0, 0, 0)
        self.assertEqual(DBCE_CLASS_CONTENT, classify_block(versions, bits))

    def test_classify_cur_up_down_10100(self):
        versions = (1, 1, 1, 0, 0)
        bits = (1, 0, 1, 0, 0)
        self.assertEqual(DBCE_CLASS_BOILER_PLATE, classify_block(versions, bits))

    def test_classify_cur_up_down_11000(self):
        versions = (1, 1, 1, 0, 0)
        bits = (1, 1, 0, 0, 0)
        self.assertEqual(DBCE_CLASS_BOILER_PLATE, classify_block(versions, bits))

    def test_classify_cur_up_down_11100(self):
        versions = (1, 1, 1, 0, 0)
        bits = (1, 1, 1, 0, 0)
        self.assertEqual(DBCE_CLASS_BOILER_PLATE, classify_block(versions, bits))

    def test_classify_cur_up_next_10000(self):
        versions = (1, 1, 0, 0, 1)
        bits = (1, 0, 0, 0, 0)
        self.assertEqual(DBCE_CLASS_BOILER_PLATE, classify_block(versions, bits))

    def test_classify_cur_up_next_10001(self):
        versions = (1, 1, 0, 0, 1)
        bits = (1, 0, 0, 0, 1)
        self.assertEqual(DBCE_CLASS_CONTENT, classify_block(versions, bits))

    def test_classify_cur_up_next_11000(self):
        versions = (1, 1, 0, 0, 1)
        bits = (1, 1, 0, 0, 0)
        self.assertEqual(DBCE_CLASS_BOILER_PLATE, classify_block(versions, bits))

    def test_classify_cur_up_next_11001(self):
        versions = (1, 1, 0, 0, 1)
        bits = (1, 1, 0, 0, 1)
        self.assertEqual(DBCE_CLASS_BOILER_PLATE, classify_block(versions, bits))

    def test_classify_cur_up_prev_10000(self):
        versions = (1, 1, 0, 1, 0)
        bits = (1, 0, 0, 0, 0)
        self.assertEqual(DBCE_CLASS_BOILER_PLATE, classify_block(versions, bits))

    def test_classify_cur_up_prev_10010(self):
        versions = (1, 1, 0, 1, 0)
        bits = (1, 0, 0, 1, 0)
        self.assertEqual(DBCE_CLASS_CONTENT, classify_block(versions, bits))

    def test_classify_cur_up_prev_11000(self):
        versions = (1, 1, 0, 1, 0)
        bits = (1, 1, 0, 0, 0)
        self.assertEqual(DBCE_CLASS_BOILER_PLATE, classify_block(versions, bits))

    def test_classify_cur_up_prev_11010(self):
        versions = (1, 1, 0, 1, 0)
        bits = (1, 1, 0, 1, 0)
        self.assertEqual(DBCE_CLASS_BOILER_PLATE, classify_block(versions, bits))

    def test_classify_cur_up_prev_next_10000(self):
        versions = (1, 1, 0, 1, 1)
        bits = (1, 0, 0, 0, 0)
        self.assertEqual(DBCE_CLASS_BOILER_PLATE, classify_block(versions, bits))

    def test_classify_cur_up_prev_next_10001(self):
        versions = (1, 1, 0, 1, 1)
        bits = (1, 0, 0, 0, 1)
        self.assertEqual(DBCE_CLASS_BOILER_PLATE, classify_block(versions, bits))

    def test_classify_cur_up_prev_next_10010(self):
        versions = (1, 1, 0, 1, 1)
        bits = (1, 0, 0, 1, 0)
        self.assertEqual(DBCE_CLASS_BOILER_PLATE, classify_block(versions, bits))

    def test_classify_cur_up_prev_next_10011(self):
        versions = (1, 1, 0, 1, 1)
        bits = (1, 0, 0, 1, 1)
        self.assertEqual(DBCE_CLASS_CONTENT, classify_block(versions, bits))

    def test_classify_cur_up_prev_next_11000(self):
        versions = (1, 1, 0, 1, 1)
        bits = (1, 1, 0, 0, 0)
        self.assertEqual(DBCE_CLASS_BOILER_PLATE, classify_block(versions, bits))

    def test_classify_cur_up_prev_next_11001(self):
        versions = (1, 1, 0, 1, 1)
        bits = (1, 1, 0, 0, 1)
        self.assertEqual(DBCE_CLASS_BOILER_PLATE, classify_block(versions, bits))

    def test_classify_cur_up_prev_next_11010(self):
        versions = (1, 1, 0, 1, 1)
        bits = (1, 1, 0, 1, 0)
        self.assertEqual(DBCE_CLASS_BOILER_PLATE, classify_block(versions, bits))

    def test_classify_cur_up_prev_next_11011(self):
        versions = (1, 1, 0, 1, 1)
        bits = (1, 1, 0, 1, 0)
        self.assertEqual(DBCE_CLASS_BOILER_PLATE, classify_block(versions, bits))

    def test_classify_cur_up_down_next_10000(self):
        versions = (1, 1, 1, 0, 1)
        bits = (1, 0, 0, 0, 0)
        self.assertEqual(DBCE_CLASS_INVALID, classify_block(versions, bits))

    def test_classify_cur_up_down_next_10001(self):
        versions = (1, 1, 1, 0, 1)
        bits = (1, 0, 0, 0, 1)
        self.assertEqual(DBCE_CLASS_CONTENT, classify_block(versions, bits))

    def test_classify_cur_up_down_next_10100(self):
        versions = (1, 1, 1, 0, 1)
        bits = (1, 0, 1, 0, 0)
        self.assertEqual(DBCE_CLASS_INVALID, classify_block(versions, bits))

    def test_classify_cur_up_down_next_10101(self):
        versions = (1, 1, 1, 0, 1)
        bits = (1, 0, 1, 0, 1)
        self.assertEqual(DBCE_CLASS_INVALID, classify_block(versions, bits))

    def test_classify_cur_up_down_next_11000(self):
        versions = (1, 1, 1, 0, 1)
        bits = (1, 1, 0, 0, 0)
        self.assertEqual(DBCE_CLASS_INVALID, classify_block(versions, bits))

    def test_classify_cur_up_down_next_11001(self):
        versions = (1, 1, 1, 0, 1)
        bits = (1, 1, 0, 0, 1)
        self.assertEqual(DBCE_CLASS_INVALID, classify_block(versions, bits))

    def test_classify_cur_up_down_next_11100(self):
        versions = (1, 1, 1, 0, 1)
        bits = (1, 1, 1, 0, 0)
        self.assertEqual(DBCE_CLASS_INVALID, classify_block(versions, bits))

    def test_classify_cur_up_down_next_11101(self):
        versions = (1, 1, 1, 0, 1)
        bits = (1, 1, 1, 0, 1)
        self.assertEqual(DBCE_CLASS_BOILER_PLATE, classify_block(versions, bits))

    def test_classify_cur_up_down_prev_combo(self):
        versions = (1, 1, 1, 1, 0)

        bits = (1, 0, 0, 0, 0)
        self.assertEqual(DBCE_CLASS_INVALID, classify_block(versions, bits))

        bits = (1, 0, 0, 1, 0)
        self.assertEqual(DBCE_CLASS_CONTENT, classify_block(versions, bits))

        bits = (1, 0, 1, 0, 0)
        self.assertEqual(DBCE_CLASS_INVALID, classify_block(versions, bits))

        bits = (1, 0, 1, 1, 0)
        self.assertEqual(DBCE_CLASS_INVALID, classify_block(versions, bits))

        bits = (1, 1, 0, 0, 0)
        self.assertEqual(DBCE_CLASS_INVALID, classify_block(versions, bits))

        bits = (1, 1, 0, 1, 0)
        self.assertEqual(DBCE_CLASS_INVALID, classify_block(versions, bits))

        bits = (1, 1, 1, 0, 0)
        self.assertEqual(DBCE_CLASS_INVALID, classify_block(versions, bits))

        bits = (1, 1, 1, 1, 0)
        self.assertEqual(DBCE_CLASS_BOILER_PLATE, classify_block(versions, bits))


    def test_classify_cur_up_down_prev_next_combo(self):
        versions = (1, 1, 1, 1, 1)

        bits = (1, 0, 0, 0, 0)
        self.assertEqual(DBCE_CLASS_INVALID, classify_block(versions, bits))

        bits = (1, 0, 0, 0, 1)
        self.assertEqual(DBCE_CLASS_INVALID, classify_block(versions, bits))

        bits = (1, 0, 0, 1, 0)
        self.assertEqual(DBCE_CLASS_INVALID, classify_block(versions, bits))

        bits = (1, 0, 0, 1, 1)
        self.assertEqual(DBCE_CLASS_CONTENT, classify_block(versions, bits))

        bits = (1, 0, 1, 0, 0)
        self.assertEqual(DBCE_CLASS_INVALID, classify_block(versions, bits))

        bits = (1, 0, 1, 0, 1)
        self.assertEqual(DBCE_CLASS_INVALID, classify_block(versions, bits))

        bits = (1, 0, 1, 1, 0)
        self.assertEqual(DBCE_CLASS_INVALID, classify_block(versions, bits))

        bits = (1, 0, 1, 1, 1)
        self.assertEqual(DBCE_CLASS_INVALID, classify_block(versions, bits))

        bits = (1, 1, 0, 0, 0)
        self.assertEqual(DBCE_CLASS_INVALID, classify_block(versions, bits))

        bits = (1, 1, 0, 0, 1)
        self.assertEqual(DBCE_CLASS_INVALID, classify_block(versions, bits))

        bits = (1, 1, 0, 1, 0)
        self.assertEqual(DBCE_CLASS_INVALID, classify_block(versions, bits))

        bits = (1, 1, 0, 1, 1)
        self.assertEqual(DBCE_CLASS_INVALID, classify_block(versions, bits))

        bits = (1, 1, 1, 0, 0)
        self.assertEqual(DBCE_CLASS_INVALID, classify_block(versions, bits))

        bits = (1, 1, 1, 0, 1)
        self.assertEqual(DBCE_CLASS_INVALID, classify_block(versions, bits))

        bits = (1, 1, 1, 1, 0)
        self.assertEqual(DBCE_CLASS_INVALID, classify_block(versions, bits))

        bits = (1, 1, 1, 1, 1)
        self.assertEqual(DBCE_CLASS_BOILER_PLATE, classify_block(versions, bits))

class TestComponents(unittest.TestCase):

    def test_ratio_001(self):
        self.assertEqual(41.0, ratio(41, 0))
        self.assertEqual(0.5, ratio(10, 20))
        self.assertEqual(3.0, ratio(150, 50))

    def test_percentage_001(self):
        self.assertEqual(49, percentage(49, 100))
        self.assertEqual(33.33333333, percentage(50, 150))
        self.assertEqual(1500, percentage(150, 10))

        self.assertEqual(10, percentage(1, 10))

    def test_DBCEMethodLSH_load_001(self):
        with self.assertRaises(TypeError):
            dm = DBCEMethodLSH()

    def test_CrossLSH_001(self):
        cl = CrossLSH()
        cl.cur = True
        cl.simup = True
        cl.prev = True

        self.assertEqual(cl.to_bits(), (1,1,0,1,0))

        cl.simdown = True
        cl.next = True
        self.assertEqual(cl.to_bits(), (1,1,1,1,1))

    def test_WarcIDX_list_001(self):
        widx = WarcIDX(["warc/dbce-heise.cdx.gz"])
        self.assertEqual(len([ _ for _ in widx.index]), 24)

    def test_WarcIDX_string_002(self):
        widx = WarcIDX("warc/dbce-wired.cdx.gz")
        self.assertEqual(len([ _ for _ in widx.index]), 12)

    def test_WarcIDX_file_not_found(self):
        widx = WarcIDX("warc/thkr_holy-20160921224209.cdx.gz1")
        with self.assertRaises(FileNotFoundError):
            self.assertEqual(len([ _ for _ in widx.index]), 15)

    def test_WarcIDX_wrong_sources(self):
        with self.assertRaises(ValueError):
            widx = WarcIDX({"warc/thkr_holy-20160921224209.cdx.gz1": True})


class TestExperimentAll(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        if not SLOW_TESTS:
            return
        # the order of the list should't have any influance on the method
        cls.warc_files = ['warc/dbce-heise.cdx.gz',
                          'warc/dbce-spiegel.cdx.gz',
                          'warc/dbce-wired.cdx.gz',
                          'warc/dbce-chefkoch.cdx.gz']

        cls.journals = idx_to_journals(config, cls.warc_files)

    @unittest.skipUnless(SLOW_TESTS, "slow experiment")
    def test_thesis(self):
        workdir = get_working_dir()
        for testname in ['T001', 'T002', 'T003', 'T004', 'T005', 'T006']:
            for domain in sorted(self.journals):
                journal = self.journals[domain]
                maschine = MaschineLSH(config, journal, outdir=workdir, testname=testname)
                maschine.process()

class TestExperimentHeise(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.test_domain = 'www.heise.de'

        # the order of the list should't have any influance on the method
        widx = WarcIDX(['warc/dbce-heise.cdx.gz'])

        cls.journal = DBCEJournal(config, domain='www.heise.de')
        # create journal
        for i in widx.index:
            cls.journal.add_item(i)

    def test_journal_001(self):

        test_domain = self.test_domain

        test_path = config.get(test_domain, 'check_path1')
        test_ts = config.getint(test_domain, 'check_ts11')
        test_wlid = config.get(test_domain, 'check_uid11')

        workdir = get_working_dir()
        maschine = MaschineLSH(config, self.journal, outdir=workdir)

        self.assertEqual(test_domain, self.journal.domain)
        journal_entry = maschine.journal.get_entry(test_path, test_ts)
        self.assertEqual(test_wlid, journal_entry['warc_entry'].wlid)

        cross = maschine.get_cross(journal_entry)
        self.assertEqual(
            config.get(test_domain, 'check_cross11'),
            cross.get_unique_id()
        )

        self.assertEqual((1, 1, 1, 1, 1), cross.to_bits())
        self.assertEqual(test_wlid, cross.cur['warc_entry'].wlid)

        diffmethod = DBCEMethodLSH(config, cross)

    def test_journal_002(self):

        test_domain = self.test_domain
        test_path = config.get(test_domain, 'check_path1')
        test_ts = config.getint(test_domain, 'check_ts12')
        test_wlid = config.get(test_domain, 'check_uid12')

        workdir = get_working_dir()
        maschine = MaschineLSH(config, self.journal, outdir=workdir)

        self.assertEqual(test_domain, self.journal.domain)
        journal_entry = maschine.journal.get_entry(test_path, test_ts)
        self.assertEqual(test_wlid, journal_entry['warc_entry'].wlid)

        cross = maschine.get_cross(journal_entry)
        self.assertEqual(
            config.get(test_domain, 'check_cross12'),
            cross.get_unique_id()
        )

        self.assertEqual((1, 1, 1, 1, 0), cross.to_bits())
        self.assertEqual(test_wlid, cross.cur['warc_entry'].wlid)

        diffmethod = DBCEMethodLSH(config, cross)

    def test_journal_003(self):

        test_domain = self.test_domain
        test_path = config.get(test_domain, 'check_path2')
        test_ts = config.getint(test_domain, 'check_ts21')
        test_wlid = config.get(test_domain, 'check_uid21')

        workdir = get_working_dir()
        maschine = MaschineLSH(config, self.journal, outdir=workdir)

        self.assertEqual(test_domain, self.journal.domain)
        journal_entry = maschine.journal.get_entry(test_path, test_ts)
        self.assertEqual(test_wlid, journal_entry['warc_entry'].wlid)

        cross = maschine.get_cross(journal_entry)
        self.assertEqual(
            config.get(test_domain, 'check_cross21'),
            cross.get_unique_id()
        )

        self.assertEqual((1, 1, 1, 0, 1), cross.to_bits())
        self.assertEqual(test_wlid, cross.cur['warc_entry'].wlid)

        diffmethod = DBCEMethodLSH(config, cross)

    def test_journal_004(self):
        test_domain = self.test_domain
        test_path = config.get(test_domain, 'check_path2')
        test_ts = config.getint(test_domain, 'check_ts22')
        test_wlid = config.get(test_domain, 'check_uid22')

        workdir = get_working_dir()
        maschine = MaschineLSH(config, self.journal, outdir=workdir)

        self.assertEqual(test_domain, self.journal.domain)
        journal_entry = maschine.journal.get_entry(test_path, test_ts)
        self.assertEqual(test_wlid, journal_entry['warc_entry'].wlid)

        cross = maschine.get_cross(journal_entry)
        self.assertEqual(
            config.get(test_domain, 'check_cross22'),
            cross.get_unique_id()
        )

        self.assertEqual((1, 1, 1, 1, 1), cross.to_bits())
        self.assertEqual(test_wlid, cross.cur['warc_entry'].wlid)

        diffmethod = DBCEMethodLSH(config, cross)

    def test_process_001(self):
        test_domain = self.test_domain

        test_path = config.get(test_domain, 'check_path1')
        test_ts = config.getint(test_domain, 'check_ts11')
        test_wlid = config.get(test_domain, 'check_uid11')

        workdir = get_working_dir()
        for testname in ['T001', 'T002', 'T006']:
            maschine = MaschineLSH(config, self.journal, outdir=workdir, testname=testname)
            maschine.process()

class TestExperimentSpiegel(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.test_domain = 'www.spiegel.de'

        # the order of the list should't have any influance on the method
        widx = WarcIDX(['warc/dbce-spiegel.cdx.gz'])

        cls.journal = DBCEJournal(config, domain=cls.test_domain)
        # create journal
        for i in widx.index:
            cls.journal.add_item(i)

    def test_journal_001(self):
        test_domain = self.test_domain

        test_path = config.get(test_domain, 'check_path1')
        test_ts = config.getint(test_domain, 'check_ts11')
        test_wlid = config.get(test_domain, 'check_uid11')

        workdir = get_working_dir()
        maschine = MaschineLSH(config, self.journal, outdir=workdir)

        self.assertEqual(test_domain, self.journal.domain)
        journal_entry = maschine.journal.get_entry(test_path, test_ts)
        self.assertEqual(test_wlid, journal_entry['warc_entry'].wlid)

        cross = maschine.get_cross(journal_entry)
        self.assertEqual(
            config.get(test_domain, 'check_cross11'),
            cross.get_unique_id()
        )

        self.assertEqual((1, 1, 0, 1, 1), cross.to_bits())
        self.assertEqual(test_wlid, cross.cur['warc_entry'].wlid)

        diffmethod = DBCEMethodLSH(config, cross)

    def test_journal_002(self):
        test_domain = self.test_domain
        test_path = config.get(test_domain, 'check_path1')
        test_ts = config.getint(test_domain, 'check_ts12')
        test_wlid = config.get(test_domain, 'check_uid12')

        workdir = get_working_dir()
        maschine = MaschineLSH(config, self.journal, outdir=workdir)

        self.assertEqual(test_domain, self.journal.domain)
        journal_entry = maschine.journal.get_entry(test_path, test_ts)
        self.assertEqual(test_wlid, journal_entry['warc_entry'].wlid)

        cross = maschine.get_cross(journal_entry)
        self.assertEqual(
            config.get(test_domain, 'check_cross12'),
            cross.get_unique_id()
        )

        self.assertEqual((1, 1, 0, 1, 0), cross.to_bits())
        self.assertEqual(test_wlid, cross.cur['warc_entry'].wlid)

        diffmethod = DBCEMethodLSH(config, cross)

    def test_journal_003(self):
        test_domain = self.test_domain
        test_path = config.get(test_domain, 'check_path2')
        test_ts = config.getint(test_domain, 'check_ts21')
        test_wlid = config.get(test_domain, 'check_uid21')

        workdir = get_working_dir()
        maschine = MaschineLSH(config, self.journal, outdir=workdir)

        self.assertEqual(test_domain, self.journal.domain)
        journal_entry = maschine.journal.get_entry(test_path, test_ts)
        self.assertEqual(test_wlid, journal_entry['warc_entry'].wlid)

        cross = maschine.get_cross(journal_entry)
        self.assertEqual(
            config.get(test_domain, 'check_cross21'),
            cross.get_unique_id()
        )

        self.assertEqual((1, 1, 0, 1, 0), cross.to_bits())
        self.assertEqual(test_wlid, cross.cur['warc_entry'].wlid)

        diffmethod = DBCEMethodLSH(config, cross)

    def test_journal_004(self):

        test_domain = self.test_domain
        test_path = config.get(test_domain, 'check_path2')
        test_ts = config.getint(test_domain, 'check_ts22')
        test_wlid = config.get(test_domain, 'check_uid22')

        workdir = get_working_dir()
        maschine = MaschineLSH(config, self.journal, outdir=workdir)

        self.assertEqual(test_domain, self.journal.domain)
        journal_entry = maschine.journal.get_entry(test_path, test_ts)
        self.assertEqual(test_wlid, journal_entry['warc_entry'].wlid)

        cross = maschine.get_cross(journal_entry)
        self.assertEqual(
            config.get(test_domain, 'check_cross22'),
            cross.get_unique_id()
        )

        self.assertEqual((1, 1, 0, 1, 1), cross.to_bits())
        self.assertEqual(test_wlid, cross.cur['warc_entry'].wlid)

        diffmethod = DBCEMethodLSH(config, cross)

if __name__ == '__main__':
    os.environ['DBCE_RESULT_SUFFIX'] = 'unittest'
    unittest.main()
