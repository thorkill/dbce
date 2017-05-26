"""
Collection of utility funtions.

Copyright (c) 2016 Rafal Lesniak

This software is licensed as described in the file LICENSE.

"""

import pdb
import datetime

import os.path
from os import (mkdir, getenv)
from os import stat as os_stat

from collections import OrderedDict
from itertools import islice

import bs4.element

from dbce.datatypes.node import NodeInfo


COLUMNS_CLF_VECTOR = ['docid',
                      'testname',
                      'domain',
                      'nodeid',
                      'dbce_class',
                      'perv_tag_id',
                      'cur_tag_id',
                      'next_tag_id',
                      'cur_class_hash',
                      'prev_ld', 'prev_td',
                      'cur_ld', 'cur_td',
                      'next_ld', 'next_td',
                      'prev_wc', 'cur_wc', 'next_wc',
                      'prev_tl', 'cur_tl', 'next_tl',
                      'prev_tc', 'cur_tc', 'next_tc',
                      'cur_t2t_ratio',
                      'l1_perv_tag_id',
                      'l1_cur_tag_id',
                      'l1_next_tag_id',
                      'l1_cur_class_hash',
                      'l1_prev_ld', 'l1_prev_td',
                      'l1_cur_ld', 'l1_cur_td',
                      'l1_next_ld', 'l1_next_td',
                      'l1_prev_wc', 'l1_cur_wc', 'l1_next_wc',
                      'l1_prev_tl', 'l1_cur_tl', 'l1_next_tl',
                      'l1_prev_tc', 'l1_cur_tc', 'l1_next_tc',
                      'l1_cur_t2t_ratio',
                      'l2_perv_tag_id',
                      'l2_cur_tag_id',
                      'l2_next_tag_id',
                      'l2_cur_class_hash',
                      'l2_prev_ld', 'l2_prev_td',
                      'l2_cur_ld', 'l2_cur_td',
                      'l2_next_ld', 'l2_next_td',
                      'l2_prev_wc', 'l2_cur_wc', 'l2_next_wc',
                      'l2_prev_tl', 'l2_cur_tl', 'l2_next_tl',
                      'l2_prev_tc', 'l2_cur_tc', 'l2_next_tc',
                      'l2_cur_t2t_ratio']

def load_groundtruth_for_cross(config, gtname):
    with open(os.path.join(get_groudtruth_dir(config), gtname), 'r') as grfd:
        for _ in grfd:
            _ = _.strip()
            (hashval, k, v) = _.split('\t', 2)
            hashval = int(hashval)
            yield (hashval, k, v)

def get_groudtruth_dir(config):
    return config.get('DEFAULT', 'groundtruth_dir')

def get_working_dir():
    workdir = getenv('DBCE_WORKDIR', None)
    resultsuf = getenv('DBCE_RESULT_SUFFIX', 'default')

    if not workdir:
        raise RuntimeError("env DBCE_WORKDIR not defined ... aborting")

    return os.path.join(workdir,
                        "{}_{}_result".format(
                            datetime.date.strftime(datetime.datetime.now(),
                                                   "%Y%m%d-%H%M%S"),
                            resultsuf))

def create_working_dir(outdir):
    try:
        os_stat(outdir)
    except FileNotFoundError:
        mkdir(outdir)

def is_tag(todo, tag_names=None, any_tag=False):
    if isinstance(todo, bs4.element.Tag):
        if (not any_tag and todo.name in tag_names) or any_tag:
            return True
    return False

def is_string(todo):
    return isinstance(todo, bs4.element.NavigableString)

def is_list_equal(lst):
    return not lst or lst.count(lst[0]) == len(lst)

def filter_tag(elements, check=None):
    for element in elements:
        if check and is_tag(element, check):
            yield element
        elif check is None and is_tag(element, any_tag=True) and 'dbce-intern' not in element.attrs:
            yield element

def filter_string(elements):
    for element in elements:
        if is_string(element):
            yield element

def ratio(x, y):
    """
    x - number of words
    y - number of tags
    """
    if y == 0:
        y = 1
    return round(x/y, 8)

def dfs(tree):
    level = 0
    level_nodes = OrderedDict()
    dfs1(tree, level, level_nodes)
    return level_nodes

def dfs1(tree, level, level_nodes):
    if is_string(tree):
        return

    _lvl_nodes = []
    for node in tree.children:
        dfs1(node, level + 1, level_nodes)
        if is_tag(node, any_tag=True):
            node.dbce_layer = level
            _lvl_nodes.insert(0, node)
    _ = level_nodes.setdefault(level, [])
    _.insert(0, _lvl_nodes)
    return

def get_layers(soup):
    return dfs(soup)

def _write_csv(outfile, header, data, auto_index=True, append=False):
    if isinstance(header, tuple) or isinstance(header, list):
        header = ",".join(header)

    if append:
        _mode = "a"
    else:
        _mode = "w"

    try:
        os.stat(outfile)
        _skip_header = True
    except FileNotFoundError:
        _skip_header = False

    with open(outfile, _mode) as out:
        _idx = 0
        if not _skip_header:
            if auto_index:
                out.write(",")
            out.write("{}\n".format(header))

        for line in data:
            if auto_index:
                out.write("{},{}\n".format(_idx, line))
                _idx += 1
            else:
                out.write("{}\n".format(line))

def dump_context_info(ctx, cross, prefix):
    columns_ctx = ['docid',
                   'domain',
                   'classifier',
                   'testname',
                   'time_analysis_end',
                   'time_analysis_start',
                   'time_dbce_method_end',
                   'time_dbce_method_start',
                   'time_s2_s_annotate_end',
                   'time_s2_s_annotate_start',
                   'time_s2_s_classify_end',
                   'time_s2_s_classify_pp_end',
                   'time_s2_s_classify_pp_start',
                   'time_s2_s_classify_start',
                   'time_s2_s_learn_end',
                   'time_s2_s_learn_start',
                   'time_s2_s_reclassify_end',
                   'time_s2_s_reclassify_start']

    outfile_ctx = os.path.join(ctx.output_dir, "df-ctx.csv")

    (_cross_header, _cross_values) = cross.to_csv()

    _ctx = [ctx.docid, ctx.domain, ctx.clf_name, ctx.testname]
    _ctx.extend(getattr(ctx, x, '') for x in columns_ctx[4:])
    _ctx.extend(_cross_values)

    _write_csv(outfile_ctx,
               columns_ctx + _cross_header,
               [','.join(map(str, _ctx))],
               auto_index=True,
               append=True)

def dump_nodes_info(ctx, nodestats, prefix=""):
    """Writes informations about the context and nodes
    into files for evaluation"""
    df_nodes = []
    df_clf_vec = []

    _node_header = None
    for node in nodestats.items():
        _ = node[1].to_csv()
        if not _node_header:
            _node_header = list(_[0])

        # index
        _node_tmp = _[1]
        _node_tmp = _node_tmp.split(',')
        _node_tmp.insert(0, ctx.domain)
        _node_tmp.insert(0, ctx.testname)
        _node_tmp.insert(0, ctx.docid)
        df_nodes.append(','.join(_node_tmp))

        # vectors for this node
        _vec = [ctx.docid, ctx.testname, ctx.domain, node[1].nodeid, node[1].dbce_class]
        if node[1].clf_vector:
            _vec.extend(node[1].clf_vector)
        else:
            _vec.extend(['']*(len(COLUMNS_CLF_VECTOR)-5))
        df_clf_vec.append(str(",".join(map(str, _vec))))

    outfile_vec = os.path.join(ctx.output_dir, "df-node-vec.csv")
    _write_csv(outfile_vec,
               COLUMNS_CLF_VECTOR,
               df_clf_vec,
               append=True)

    outfile = os.path.join(ctx.output_dir, "df-nodes.csv")
    _write_csv(outfile, ['docid', 'testname', 'domain'] + _node_header, df_nodes,
               append=True)

def get_node_info(node, nodestats, no_insert=False):
    n_id = id(node)
    try:
        return nodestats[n_id]
    except KeyError:
        if not no_insert:
            return nodestats.setdefault(n_id, NodeInfo(node))
        else:
            raise ValueError("node {} should be defined".format(n_id))


def get_element(node):
    # for XPATH we have to count only for nodes with same type!
    length = len(list(node.previous_siblings)) + 1
    if (length) > 1:
        return '%s:nth-child(%s)' % (node.name, length)
    else:
        return node.name

def get_css_path(node):
    path = [get_element(node)]
    for parent in node.parents:
        if parent.name == 'body':
            break
        path.insert(0, get_element(parent))
    return ' > '.join(path)

def get_xpath(element, ctx):
    assert isinstance(element, bs4.element.Tag)

    el_info = get_node_info(element, ctx.nodestats)
    if not el_info.xpath:
        el_info.xpath = get_css_path(element)

    return el_info.xpath

def __get_xpath(element, ctx):
    assert isinstance(element, bs4.element.Tag)

    el_info = get_node_info(element, ctx.nodestats)
    if el_info.xpath:
        return el_info.xpath

    components = []
    child = element

    for parent in child.parents:
        pc_idx = {}
        pc_idx_c = 0
        for _ in parent.contents:
            pc_idx[id(_)] = pc_idx_c
            pc_idx_c += 1

        previous = islice(parent.children, 0, pc_idx[id(child)])

        xpath_tag = child.name
        if get_node_info(child, ctx.nodestats).node_name not in ['html', 'body', 'dbce'] \
           and 'class' in child.attrs:
            _classes = child['class'] if isinstance(child['class'], list) else [child['class']]
            for cls in _classes:
                xpath_tag = "{}.{}".format(xpath_tag, cls)

        xpath_index = sum(1 for i in previous if i.name == xpath_tag) + 1
        components.append(xpath_tag if xpath_index == 1 else '%s:nth-of-type(%d)' % (xpath_tag, xpath_index))
        child = parent
    components.reverse()
    el_info.xpath = ' > '.join(components)
    el_info.xpath_hash = hash(el_info.xpath)
    return el_info.xpath
