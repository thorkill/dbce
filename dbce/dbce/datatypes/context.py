
"""
DBCECTX - context container for one cross

Copyright (c) 2016 Rafal Lesniak. All rights reserved.

This software is licensed as described in the file LICENSE.
"""

from dbce.datatypes.node import NodeStats

class DBCECTX(object):

    def __init__(self, config, wlid, testname=None):
        self.docid = wlid
        self.nodestats = NodeStats()
        self.seen_tags = set(['dbce-dummy'])
        self.config = config
        self.testname = testname

        self.clf_data = {}
        self.clf_name = None
        self.clf_params = None

        # timing
        self.time_start = 0
        self.time_end = 0

        self.time_analysis_end = 0
        self.time_analysis_start = 0
        self.time_dbce_method_end = 0
        self.time_dbce_method_start = 0
        self.time_analysis_end = 0
        self.time_s2_s_annotate_end = 0
        self.time_s2_s_annotate_start = 0
        self.time_s2_s_classify_end = 0
        self.time_s2_s_classify_pp_end = 0
        self.time_s2_s_classify_pp_start = 0
        self.time_s2_s_classify_start = 0
        self.time_s2_s_learn_end = 0
        self.time_s2_s_learn_start = 0
        self.time_s2_s_reclassify_end = 0
        self.time_s2_s_reclassify_start = 0
        self.time_analysis_start = 0

        # cross
        self.cross_unique_id = None

        # machine learning
        self.clf_scores = None
        self.class_marker = None
        self.clf_vectors = None

        # output files
        self.output_dir = None
        self.output_file = "{}-{}.html".format(testname, wlid)
        self.output_org_file = "{}-org-{}.html".format(testname, wlid)
        self.output_ann_file = "{}-ann-{}.html".format(testname, wlid)
        self.output_san_file = "{}-org-san-{}.html".format(testname, wlid)
        self.output_manual_annotation = "{}-{}-man-ann.txt".format(testname, wlid)
        self.output_file_merged = "{}-merged-{}.html".format(testname, wlid)
        self.output_file_txt = "{}-{}.txt".format(testname, wlid)

        self.has_confusion_matrix = False

        # analysis results
        self.bs_diff = None
        self.bs_diff_merged = None
        self.extracted_data = None
