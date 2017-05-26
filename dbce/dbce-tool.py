#!/usr/bin/env python3

import argparse

import sys
import time
import os

import configparser
config = configparser.ConfigParser()

import pprint

import itertools

import logging
import logging.config

from urllib.parse import urlparse

import pdb

import random

from dbce.WarcIDX import WarcIDX
from dbce.DBCEJournal import (DBCEJournal, idx_to_journals)
from dbce.experiment import MaschineLSH
from dbce.utils import get_working_dir

from dbce.evaluation import load_eval_results

from matplotlib import pyplot as plt

from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform

DBCE_BINDIR = os.environ.get('DBCE_BINDIR', './')

logging.config.fileConfig('{}/logging.conf'.format(DBCE_BINDIR))

# create logger
logger = logging.getLogger('dbce') # pylint: disable=C0103

PLOT_DPI = 50

if __name__ == '__main__':
    os.environ['DBCE_RESULT_SUFFIX'] = 'diff-it'
    # FIXME: make proper help/descriptions
    parser = argparse.ArgumentParser(
        description='generate diff between two files using DMP')

    parser.add_argument('--mode', choices=['idx'], default='idx')
    parser.add_argument('--idx', dest='idx_sources', nargs="+")
    parser.add_argument('--domains', dest='domain', nargs="+")

    parser.add_argument('--lsh', dest='lsh_only', action='store_true', default=False)
    parser.add_argument('--manual-annotate', dest='manual_annotate', action='store_true', default=False)
    parser.add_argument('--report', dest='report_dir')

    args = parser.parse_args()
    config.read('{}/dbce.ini'.format(DBCE_BINDIR))

    if args.mode == 'idx' and not args.idx_sources and not args.report_dir:
        parser.error('missing arguments')

    elif args.report_dir:
        try:
            erc = load_eval_results(config, args.report_dir)
        except:
            import sys
            import traceback
            atype, value, tb = sys.exc_info()
            traceback.print_exc()
            pdb.post_mortem(tb)

    elif args.mode == 'idx':
        journals = idx_to_journals(config, args.idx_sources)
        if args.lsh_only and not args.manual_annotate:
            workdir = get_working_dir()
            for domain in sorted(journals):
                journal = journals[domain]
                try:
                    maschine = MaschineLSH(config, journal, outdir=workdir)
                    maschine.process()
                except:
                    import sys
                    import traceback
                    atype, value, tb = sys.exc_info()
                    traceback.print_exc()
                    pdb.post_mortem(tb)

        elif args.manual_annotate:
            if len(args.domain) != 1:
                parser.error('manual annotation needs exactly one domain')
            workdir = get_working_dir()
            for domain in sorted(journals):
                journal = journals[domain]
                maschine = MaschineLSH(config, journal, outdir=workdir)
                maschine.manual_annotate(domains=args.domain)
