"""
MaschineLSH - the machinery behind the analysis of the datasets.

Copyright (c) 2016 Rafal Lesniak

This software is licensed as described in the file LICENSE.
"""

import datetime
import logging
import pdb
import time
import os.path
import random
from urllib.parse import urlparse
from dbce import DBCEMethod

from dbce.datatypes.context import DBCECTX

from dbce.classification import (marker_to_bits, classify_block)
from dbce.constants import (DBCE_MARKUP_ANALYSIS_TAG, DBCE_CLASS_CONTENT,
                            DBCE_CLASS_BOILER_PLATE, DBCE_CLASS_INVALID)
from dbce.utils import (get_xpath, create_working_dir,
                        get_groudtruth_dir, load_groundtruth_for_cross)
from dbce.evaluation import tune_hyper_parameter

from dbce.datatypes.cross import CrossLSH

logger = logging.getLogger(__name__)

def dump_cross_stats(_crosses, domain, workdir):
    for bits in _crosses:
        _bits = "".join(map(str, bits))
        with open(os.path.join(workdir,
                               "crosses-{}-{}.txt".format(_bits, domain)), "w") as outf:
            for record in sorted(_crosses[bits]):
                outf.write("\t".join(record))
                outf.write("\n")

def dump_result(ctx):

    outfile = os.path.join(ctx.output_dir, ctx.output_file)

    with open(outfile, "w") as fout:
        fout.write(ctx.bs_diff.prettify())

    outfile = os.path.join(ctx.output_dir, ctx.output_file_merged)
    with open(outfile, "w") as fout:
        fout.write(ctx.bs_diff_merged.prettify())

    outfile = os.path.join(ctx.output_dir, ctx.output_file_txt)
    with open(outfile, "w") as fout:
        fout.write(ctx.extracted_data)

    outfile = os.path.join(ctx.output_dir, ctx.output_san_file)
    with open(outfile, "w") as fout:
        fout.write(ctx.bs_html_sanitized.prettify())

def _sample_data_set(ctx, versions, all_data, manual_ann_data, manual_ann_cnt):
    # try to split elements evenly
    # content
    bin_class_1 = []
    # bp
    bin_class_2 = []
    # invalid
    bin_class_3 = []
    _vectors = []
    for node in all_data:
        _xpath = get_xpath(node, ctx)
        if _xpath in manual_ann_data:
            continue
        _class = classify_block(versions, marker_to_bits(node['class']))
        if _class == DBCE_CLASS_CONTENT:
            bin_class_1.append(node)
        elif _class == DBCE_CLASS_BOILER_PLATE:
            bin_class_2.append(node)
        elif _class == DBCE_CLASS_INVALID:
            bin_class_3.append(node)
        else:
            raise ValueError("unknown class: {}".format(_class))

    logger.info("got {} / {} / {} elements per class".format(len(bin_class_1),
                                                             len(bin_class_2),
                                                             len(bin_class_3)))

    n_sample = 600
    test_data = []

    diff = manual_ann_cnt[DBCE_CLASS_CONTENT] - n_sample
    if diff < 0:
        test_data += random.sample(bin_class_1, min(len(bin_class_1), abs(diff)))

    diff = manual_ann_cnt[DBCE_CLASS_BOILER_PLATE] - n_sample
    if diff < 0:
        test_data += random.sample(bin_class_2, min(len(bin_class_2), abs(diff)))

    diff = manual_ann_cnt[DBCE_CLASS_INVALID] - n_sample
    if diff < 0:
        test_data += random.sample(bin_class_3, min(len(bin_class_3), abs(diff)))
    # FIXME: remove this change
    test_data += bin_class_1 + bin_class_2 + bin_class_3
    logger.info("Test data: {}".format(len(test_data)))
    return test_data


class MaschineLSH:
    def __init__(self, config, journal, outdir, testname=None):
        self.config = config
        self.journal = journal
        self.report = None
        self.workdir = outdir
        self.testname = testname
        self.clf_data = {'agg_dataset': [],
                         'agg_datacls': [],
                         'clf_vectors': []}
        self._gt_cache = {}
        # initialize working directory
        create_working_dir(self.workdir)

    def get_similar(self, myid, minhash):
        _query_result = self.journal.lsh.query(minhash)
        # sort descending by similarity
        _simmilar_list = []
        _candidates = []
        for _ in _query_result:
            _simmilar_list.append(
                self.journal.get_by_id(_)['minhash'].jaccard(minhash)
                )

        # sort by ascending similarity
        _query_result = sorted(list(zip(_simmilar_list, _query_result)),
                               key=lambda x: x[0])

        for can in _query_result:
            # skip myself
            if can[1] == myid:
                continue
            _candidates.append(can[1])

        logger.info("candidates all: {}".format(_candidates))
        return _candidates

    def _split_candidates(self, _candidates, myurl):
        _can_hor = []
        _can_ver = []

        for can in _candidates:
            try:
                _can_entry = self.journal.get_entry_by_id(can)
            except KeyError:
                logger.error('entry id: {} not found'.format(can))
                continue

            if _can_entry['url'] == myurl:
                _can_hor.append(can)
            else:
                _can_ver.append(can)
        return (_can_ver, _can_hor)

    def _fill_best_vertical(self, _cross, _can_ver):
        # _can_ver and _can_hor are sorted by similarity
        while len(_can_ver) > 0 and not (_cross.simup and _cross.simdown):
            _tmp = self.journal.get_entry_by_id(_can_ver.pop())
            if not _cross.simup:
                _cross.simup = _tmp
                continue
            elif not _cross.simdown and \
                 _tmp['url'] != _cross.simup['url']:
                _cross.simdown = _tmp
                break

    def _fill_best_horizontal(self, _cross, _can_hor):
        while len(_can_hor) > 0:
            _tmp = self.journal.get_entry_by_id(_can_hor.pop())
            if not _cross.next and _tmp['timestamp'] > _cross.cur['timestamp']:
                _cross.next = _tmp
            elif not _cross.prev and _tmp['timestamp'] < _cross.cur['timestamp']:
                _cross.prev = _tmp
            elif _cross.prev and _tmp['timestamp'] > _cross.prev['timestamp'] \
                 and _tmp['timestamp'] < _cross.cur['timestamp']:
                logger.warning("change prev from {} to {} ({})".format(_cross.prev['timestamp'],
                                                                       _tmp['timestamp'],
                                                                       _cross.cur['timestamp']))
                _cross.prev = _tmp

            elif _cross.next and _tmp['timestamp'] < _cross.next['timestamp'] \
                 and _tmp['timestamp'] > _cross.cur['timestamp']:
                logger.warning("change next from {} to {} ({})".format(_cross.next['timestamp'],
                                                                       _tmp['timestamp'],
                                                                       _cross.cur['timestamp']))
                _cross.next = _tmp

    def get_cross(self, entry):
        """returns a CrossLSH object filled with records to for analysis

        The cross will be build based on candidates from the LSH for given
        entry."""

        _cross = CrossLSH()
        _cross.cur = entry

        logger.warning('build cross using LSH for {} ({})'.format(entry['url'], entry['timestamp']))

        # select candidates
        _candidates = self.get_similar(_cross.cur['item_id'], _cross.cur['minhash'])

        # split to vertical and horizontal
        (_can_ver, _can_hor) = self._split_candidates(_candidates, entry['url'])
        logger.info("candidates: v:{} h:{}".format(_can_ver, _can_hor))

        # fill the cross
        self._fill_best_vertical(_cross, _can_ver)
        self._fill_best_horizontal(_cross, _can_hor)

        return _cross

    def start_report(self, filename):
        """This function can be used before analysis starts.

        Create rerport file, write HTML header."""

        logger.info('Starting LSH report')

        self.report = open(filename, "w")
        self.report.write("<html><head>\n")
        self.report.write("""<style>

        h2 {{ font-size: 48px; }}
        p  {{ font-size: 1.2em; }}
        a  {{ font-size: 0.9em; }}
        table, th, td {{
          border: 1px solid black;
        }}

        td.current {{ background-color: #D8F0DA; }}
        {}
        </style></head><body>\n""".format(DBCEMethod.DBCE_STYLE))
        self.report.write('<h1> Report description </h1>')
        self.report.write('''How to read the report: each table contains nine cells which
        corresponds to the horizontal and vertical snapshots of an url.
        The cells can be read as follows:</br>''')
        self.report.write('''<a href="./simm.html">Analysis of the similarity of documents</a>
        <a href="./plan2-False.html">Manual</a> <a href="./plan2-True.html">Automatic</a></br>''')
        self.report.write('<table>\n')
        self.report.write('<col width="25%"><col width="50%"><col width="25%">\n')
        self.report.write('<tr><td>-</td><td class="diff-dir-3">simmilar link "up"</td><td>-</td></tr>\n')
        self.report.write('<tr><td class="{}">previous snapshot of analysed url'
                          '- past in time (version)</td><td class="{}"> analysed'
                          'url (version) </td><td class="{}"> next snapshot of '
                          'analysed url - further in time (version)</td></tr>\n'
                          .format(DBCEMethod.DBCE_DIFF_PREV, DBCEMethod.DBCE_DIFF_CUR,
                                  DBCEMethod.DBCE_DIFF_NEXT))
        self.report.write('<tr><td>-</td><td class="diff-dir-4">simmilar link "down"</td><td>-</td></tr>\n')
        self.report.write('</table>')

    def report_analysis(self, cross, result):
        self.report.write('<small>cross uid: {}</small>'.format(cross.get_unique_id()))
        self.report.write('<table>\n')
        self.report.write('<col width="25%"><col width="50%"><col width="25%">\n')

        if cross.simup:
            self.report.write('<tr><td></td><td>{}</td><td></td></tr>'.format(cross.simup['url']))
        else:
            self.report.write('<tr><td></td><td>-</td><td></td></tr>')

        self.report.write('<tr>')

        if cross.prev:
            self.report.write('<td>{} ({})</td>\n'.format(cross.prev['url'],
                                                          cross.prev['timestamp']))
        else:
            self.report.write('<td></td>\n')

        self.report.write('<td><a href="{}">{}</a>({})'.format(result.output_file,
                                                               cross.cur['url'],
                                                               cross.cur['timestamp']))
        self.report.write('<a href="{}">M</a> </br>'.format(result.output_file_merged))
        if result.clf_scores is not None:
            self.report.write(
                'Accuracy: {:0.2f} (+/- {:0.2f}) [{}] '.format(
                    result.clf_scores.mean(),
                    result.clf_scores.std() * 2,
                    len(result.clf_vectors)))
        self.report.write('&nbsp; <a href="{}"> content </a>'.format(result.output_file_txt))
        self.report.write('</td>\n')

        if cross.next:
            self.report.write('<td>{} ({})</td>\n'.format(cross.next['url'],
                                                          cross.next['timestamp']))
        else:
            self.report.write('<td></td>\n')
        self.report.write('</tr>')

        if cross.simdown:
            self.report.write('<tr><td></td><td>{}</td><td></td></tr>'.format(cross.simdown['url']))
        else:
            self.report.write('<tr><td></td><td>-</td><td></td></tr>')
        self.report.write('</table>\n<br>\n')
        self.report.write('<small>Timing: total: {:.3f}s / '
                          'annotate: {:.3f}s / merge : {:.3f}s / '
                          'pp: {:.3f}s / s1: {:.3f}s / learn: {:.3f}s / s3: {:.3f}s</small>'.format(
                              result.timing['time_end']-result.timing['time_start'],
                              result.timing['time_annotate_end']-result.timing['time_annotate_start'],
                              result.timing['time_merge_end']-result.timing['time_merge_start'],
                              result.timing['time_preprocess_end']-result.timing['time_preprocess_start'],
                              result.timing['time_stage1_end']-result.timing['time_stage1_start'],
                              result.timing['time_learn_end']-result.timing['time_learn_start'],
                              result.timing['time_stage3_end']-result.timing['time_stage3_start']
                          ))
        if result.has_confusion_matrix:
            self.report.write('<img src="{}" /> <img src="{}" /></br>\n'.format(
                result.output_confusion_matrix,
                result.output_confusion_matrix_norm))
        self.report.flush()

    def close_rerport(self, time_consumed=None):
        """Called at the end of the analysis.

        Close HTML report"""

        if time_consumed:
            self.report.write(
                '<div>Consumed time: {:.2f}s, on {}</div>'.format(
                    time_consumed,
                    datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
                )
            )

        self.report.write("</body></html>")
        self.report.close()

    def report_new_domain(self, domain):
        self.report.write('<h2> {} </h2>\n'.format(domain))

    def process(self):
        domain = self.journal.domain
        logger.info("processing ... {}".format(domain))
        time_start = time.time()
        report_filename = os.path.join(self.workdir, "report_lsh-{}.html".format(domain))
        self.start_report(report_filename)

        self.process_domain(domain)

        time_end = time.time()
        self.close_rerport(time_consumed=time_end-time_start)

    def process_domain(self, domain=None):
        self.report_new_domain(domain)

        _crosses = {}
        _check_crosses = self.cfg_cross(domain)
        _only_def_crosses = self.config.getboolean(domain, 'only_defined_crosses')
        _opt_tune = self.config.getboolean(domain, 'tune_hyper_parameters')

        for (j_path, j_ts) in self.journal.get_paths():
            entry = self.journal.get_entry(j_path, j_ts)
            cross = self.get_cross(entry)
            cross.window_size = self.config.getint('DEFAULT', 'compress_window_size')
            if _only_def_crosses and cross.get_unique_id() not in _check_crosses:
                continue

            logger.info("Cross bits: {}-{}".format(cross.window_size,
                                                   cross.to_bits()))
            # we need at least one similar document
            if not cross.simup:
                logger.error("Analysis need at least one similar document")
                continue

            _cross_log = _crosses.setdefault(cross.to_bits(), [])
            _cross_log.append((cross.cur['warc_entry'].wlid,
                               j_path, str(j_ts), cross.get_unique_id()))

            if cross.to_bits().count(1) < self.config.getint(domain, 'minimal_cross_size'):
                logger.error("In this research we need at least 4 cross elements")
                continue

            diffmethod = DBCEMethod.DBCEMethodLSH(self.config, cross, testname=self.testname)

            ctx = DBCECTX(self.config, cross.cur['warc_entry'].wlid, self.testname)
            ctx.domain = domain
            ctx.output_dir = self.workdir
            ctx.cross_unique_id = cross.get_unique_id()
            diffmethod.analysis(ctx)

            if _opt_tune:
                self.collect_datasets(ctx)
            dump_result(ctx)
            #self.report_analysis(cross, ctx)
            logging.info("Results stored into file://{}".format(ctx.output_dir))

        if _opt_tune:
            tune_hyper_parameter(self.config, _crosses, self.clf_data, domain, self.workdir)

        dump_cross_stats(_crosses, domain, self.workdir)

    def cfg_cross(self, domain):
        cfg_get = self.config.get
        return [cfg_get(domain, 'check_cross11'),
                cfg_get(domain, 'check_cross12'),
                cfg_get(domain, 'check_cross21'),
                cfg_get(domain, 'check_cross22')]

    def collect_datasets(self, ctx):
        self.clf_data['agg_dataset'].extend(ctx.clf_data['clf_agg_dataset'])
        self.clf_data['agg_datacls'].extend(ctx.clf_data['clf_agg_datacls'])
        for (_vec, _data) in ctx.clf_vectors.items():
            if _data[1] == DBCE_CLASS_CONTENT:
                _cls = 1
            else:
                _cls = 0
            self.clf_data['clf_vectors'].append((hash(_data[0]), _vec, _cls))

    def manual_annotate(self, domains=None):
        import tempfile
        driver = None
        tmpfile = tempfile.NamedTemporaryFile(mode="w")
        self._gt_cache = {}

        for (j_path, j_ts) in self.journal.get_paths():
            driver = self.manual_annotate_path_ts(j_path, j_ts, domains[0], driver, tmpfile)

    def _load_gt_for_cross(self, cross):
        # load groundtruth
        manual_ann_data = {}
        manual_ann_cnt = {DBCE_CLASS_CONTENT: 0,
                          DBCE_CLASS_BOILER_PLATE: 0,
                          DBCE_CLASS_INVALID: 0}

        try:
            for (_hs, k, v) in load_groundtruth_for_cross(self.config, cross.get_unique_id()):
                if k in manual_ann_data and k != manual_ann_data[v]:
                    raise ValueError
                manual_ann_data[k] = v
                if v in manual_ann_cnt:
                    manual_ann_cnt[v] += 1
        except FileNotFoundError:
            pass

        return (manual_ann_data, manual_ann_cnt)

    def manual_annotate_path_ts(self, j_path, j_ts, domain, driver=None, tmpfile=None):
        _check_crosses = self.cfg_cross(domain)
        groundtruthdir = get_groudtruth_dir(self.config)

        entry = self.journal.get_entry(j_path, j_ts)
        cross = self.get_cross(entry)
        cross.window_size = self.config.getint('DEFAULT', 'compress_window_size')
        if cross.get_unique_id() not in _check_crosses:
            logger.warning("Skip cross: {}".format(cross.get_unique_id()))
            return driver

        logger.info("Cross bits: {} {}".format(cross.to_bits(), cross.get_unique_id()))
        if cross.to_bits().count(1) < self.config.getint(domain, 'minimal_cross_size'):
            return driver

        ctx = DBCECTX(self.config, cross.cur['warc_entry'].wlid, self.testname)

        diffmethod = DBCEMethod.DBCEMethodLSH(self.config, cross)
        data_set = diffmethod.setup_dataset()

        # annotation based on cross information
        bs_data = diffmethod.stage2_step_annotate(ctx, data_set)[1]

        # classify blocks based on bits and versions
        versions = cross.to_bits()
        all_data = [_ for _ in bs_data.findAll(DBCE_MARKUP_ANALYSIS_TAG)]
        logging.info("Found {} elements".format(len(all_data)))
        cur_url_parsed = urlparse(cross.cur['url'])
        DBCEMethod.inject_header(bs_data, cur_url_parsed)

        # load groundtruth
        (manual_ann_data, manual_ann_cnt) = self._load_gt_for_cross(cross)
        for _xpath in manual_ann_data:
            if _xpath not in self._gt_cache:
                self._gt_cache[_xpath] = manual_ann_data[_xpath]

        test_data = _sample_data_set(ctx, versions, all_data,
                                     manual_ann_data, manual_ann_cnt)

        xpath_test_data = list(zip(map(lambda x: get_xpath(x, ctx), test_data), test_data))

        def dump_groundtruth(groundtruthdir, cross_unique_id, manual_ann_data):
            with open(os.path.join(groundtruthdir,
                                   cross_unique_id), "w") as gtfd:
                for (k, v) in sorted(manual_ann_data.items(), key=lambda x: x[0]):
                    gtfd.write("{:d}\t{}\t{}\n".format(hash(k), k, v))

        logger.debug(" ----> {} elements prepared for anntoation".format(len(xpath_test_data)))
        for (xpath, todo) in xpath_test_data:
            if xpath not in self._gt_cache and xpath in manual_ann_data:
                self._gt_cache[xpath] = manual_ann_data[xpath]

            if xpath in manual_ann_data:
                logging.warning(" skip xpath: {}".format(xpath))
                continue
            # FIXME: _gt_cache should be removed or at least configurable
            if xpath in self._gt_cache:
                manual_ann_data[xpath] = self._gt_cache[xpath]
                manual_ann_cnt[self._gt_cache[xpath]] += 1
                dump_groundtruth(groundtruthdir, cross.get_unique_id(), manual_ann_data)
                continue

            driver = self.manual_annotate_element(xpath, todo, bs_data, manual_ann_data, manual_ann_cnt,
                                                  driver, tmpfile)

            dump_groundtruth(groundtruthdir, cross.get_unique_id(), manual_ann_data)

        return driver

    def manual_annotate_element(self, xpath, todo, bs_data, manual_ann_data, manual_ann_cnt,
                                driver, tmpfile):
        parent = todo.parent
        done = False
        while not done:
            print("\n\n\n")
            print("#"*60)
            print("\n")
            print("# ", xpath)
            print("\n")
            print("#"*30)
            print(todo.prettify())
            tmpfile.seek(0)

            try:
                _old_style = todo['style']
            except KeyError:
                _old_style = None

            try:
                _old_id = todo['id']
            except KeyError:
                _old_id = None

            todo['style'] = "background: red; border:2px; border-style:dotted; border-color: green;"
            _tag_id = "eluvhcvm2e"
            todo['id'] = _tag_id
            tmpfile.write(bs_data.prettify())
            if not driver:
                from selenium import webdriver
                driver = webdriver.Firefox()
            driver.get("file://{}".format(tmpfile.name))
            try:
                driver.implicitly_wait(1)
                el = driver.find_element_by_id(_tag_id)
                #el.location_once_scrolled_into_view
                elh = el.location
                window_size = driver.get_window_size()
                top = elh['y']-(window_size['height']/2)
                #driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
                driver.execute_script("window.scrollTo(0,{});".format(top))
            except:
                el = None
                print("!"*100)

            todo['style'] = _old_style
            todo['id'] = _old_id

            print("\n")
            print("#"*30)
            print("# Context:\n\n\n")
            print(parent.prettify()[0:1024])
            print("\n")
            print("#"*30)
            print("\n")

            if el and not el.is_displayed():
                print("# !!!! # !!!! # !!!! I N V I S I B L E !!! # !!!! # !!!! #")
                manual_ann_data[xpath] = DBCE_CLASS_BOILER_PLATE
                manual_ann_cnt[DBCE_CLASS_BOILER_PLATE] += 1
                done = True
                continue

            try:
                print(manual_ann_cnt)
                inp = input("1/2  [ s - skip, + - more context ] content/boilerplate ... submit with enter: ")
                if inp == 's':
                    done = True
                    continue
                elif inp == '+':
                    parent = parent.parent
                    continue
                elif inp not in ['1', '2']:
                    continue

                manual_ann_data[xpath] = DBCE_CLASS_CONTENT if inp == '1' else DBCE_CLASS_BOILER_PLATE
                self._gt_cache[xpath] = manual_ann_data[xpath]
                manual_ann_cnt[manual_ann_data[xpath]] += 1
                done = True

            except KeyboardInterrupt:
                done = True
                if driver:
                    driver.quit()
        return driver
