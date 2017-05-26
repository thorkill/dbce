"""
DBCEMethod - Implementation of Stage 2 and Stage 3

Copyright (c) 2016 Rafal Lesniak

This software is licensed as described in the file LICENSE.
"""

from html import escape as html_escape
from urllib.parse import urlparse
from collections import (Counter, OrderedDict)

import copy
import time
import logging

import pdb
import textwrap

from dbce.constants import (DBCE_MARKUP_PRESENTATION_TAG, DBCE_MARKUP_ANALYSIS_TAG,
                            DBCE_CLASS_UNKNWON, DBCE_CLASS_BOILER_PLATE,
                            DBCE_CLASS_CONTENT, DBCE_CLASS_INVALID,
                            DBCE_DIFF_CUR, DBCE_DIFF_UP, DBCE_DIFF_DOWN,
                            DBCE_DIFF_PREV, DBCE_DIFF_NEXT,
                            DBCE_STYLE, DBCE_VALID_TAGS)

from dbce.dbcediff import html_annotate
from dbce.datatypes.node import (NodeInfo, NodeStats)

from dbce.classification import (marker_to_bits, classify_block)

from dbce.utils import (get_layers, is_tag, is_string, is_list_equal,
                        dump_nodes_info, dump_context_info,
                        filter_tag, ratio, get_xpath, get_node_info)

import bs4.element
from bs4 import BeautifulSoup

from sklearn import (tree, svm)
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

logger = logging.getLogger(__name__)
DUMMY_NODE = bs4.element.Tag(name='dbce-dummy')

def fix_urls(soup, url_parsed):
    """ Fixes urls in HTML document so it can be viewed locally.

    This function changes the location of several resources if the it's link
    is relative (//something.css).

    Elements beeing changed:
    - link to stylesheet and script
    - img location
    """
    logger.info("fixing urls")
    for node in soup.find_all('link'):
        if 'rel' in node.attrs and node.attrs['rel'][0] in ['stylesheet', 'script'] and \
           node.attrs['href'][0] == '/':
            if node.attrs['href'][0:2] == '//':
                _url = "{}:".format(url_parsed.scheme)
            else:
                _url = "{}://{}".format(url_parsed.scheme, url_parsed.hostname)
            node.attrs['href'] = "{}{}".format(_url, node.attrs['href'])

    for node in soup.find_all('img'):
        if 'src' in node.attrs and node.attrs['src'][0] == '/':
            if node.attrs['src'][0:2] == '//':
                _url = "{}:".format(url_parsed.scheme)
            else:
                _url = "{}://{}".format(url_parsed.scheme, url_parsed.hostname)
            node.attrs['src'] = "{}{}".format(_url, node.attrs['src'])

    return soup

def inject_header(soup, cur_url_parsed):
    logger.info("inject header started")
    fix_urls(soup, cur_url_parsed)

    style = bs4.element.Tag(name='style')
    style_text = bs4.element.NavigableString(DBCE_STYLE)
    style['type'] = "text/css"
    style.append(style_text)
    soup.head.append(style)
    # FIXME: here we should embedd the script into the page
    script = bs4.element.Tag(name='script')
    script['src'] = "dbce.js"
    soup.head.append(script)

def dbce_markup(text, version):
    """ Return XML annotated text as string.

    text: to annotate (str)
    version: a list of elements to write into class

    Example:
    >>> dbce_markup('this is my text', ['v1', 'v2'])
    <dbce class="v1 v2">this is my text</dbce>
    """

    assert isinstance(version, list)
    _version = " ".join(version)
    return '<dbce class="%s">%s</dbce>' % (
        html_escape(str(_version), 1), text)

def _insert_class_value(todo, cls_val):
    """ insert new class value to existing class"""
    assert isinstance(todo['class'], list)
    assert cls_val
    todo['class'].append(cls_val)

def _copy_class_values(left, right):
    """ copy class values from left object to right object"""
    assert 'class' in right.attrs

    if isinstance(left['class'], str):
        _to_copy = [left['class']]
    else:
        _to_copy = left['class']

    assert isinstance(_to_copy, list)

    for cls_val in _to_copy:
        if cls_val not in right['class']:
            right['class'].append(cls_val)

def extend_nodes_class(node, ctx):
    if 'class' not in node.attrs:
        node.attrs['class'] = []
    node_info = get_node_info(node, ctx.nodestats, no_insert=True)

    _insert_class_value(node, node_info.dbce_class)
    _insert_class_value(node, 'dbce-marker')
    assert None not in getattr(node.attrs, 'class', [])

def wrap_with_div(soup, ctx, versions=None, clean_markup=False, use_dbce=True):
    """ Retruns BeautifulSoup object in which the <dbce></dbce> tags
    has been wrapped with <div><dbce></dbce></div>. If clean_markup is True
    then <dbce></dbce> tags will be removed"""

    if versions:
        _new_tag = soup.new_tag('dbce-info')
        _new_tag['class'] = []
        _bits = [_ for _ in map(str, versions)]
        _insert_class_value(_new_tag, "dbce-bits-{}".format(''.join(_bits)))
        soup.body.insert(0, _new_tag)

    for todo in soup.find_all(DBCE_MARKUP_ANALYSIS_TAG):
        todo_info = get_node_info(todo, ctx.nodestats, no_insert=True)

        _new_tag = soup.new_tag(DBCE_MARKUP_PRESENTATION_TAG)
        _new_tag['class'] = []
        _node_bits = marker_to_bits(todo['class'])
        if use_dbce:
            _content_class = DBCE_CLASS_UNKNWON
            if versions:
                _content_class = classify_block(versions, _node_bits)
            if _content_class == DBCE_CLASS_UNKNWON:
                _copy_class_values(todo, _new_tag)
        else:
            if todo_info.dbce_class is not None:
                _content_class = todo_info.dbce_class
                _insert_class_value(_new_tag, 'dbce-no-bits')
            else:
                raise ValueError("{}.dbce_class should be defined at this stage".format(todo_info.nodeid))

        _insert_class_value(_new_tag, _content_class)
        _insert_class_value(_new_tag, 'dbce-marker')

        _insert_class_value(_new_tag, "dbce-bits-{}".format(''.join(map(str, _node_bits))))
        todo.wrap(_new_tag)

        if clean_markup:
            todo.unwrap()

    return soup

def is_content(tag):
    assert isinstance(tag, NodeInfo)
    return tag.dbce_class == DBCE_CLASS_CONTENT

def _init_stats(ctx, soup):
    nodestats = ctx.nodestats
    layers = get_layers(soup)
    for layer in sorted(layers, reverse=True):
        for nodes in layers[layer]:
            for node in nodes:
                if isinstance(node, bs4.element.Tag):
                    ctx.seen_tags.add(node.name)
                    get_node_info(node, nodestats)

def percentage(part, whole):
    if whole == 0:
        whole = 0.0000001
    return round(100 * float(part)/float(whole), 8)

def extend_nodes_info(ctx, element):
    node_info = get_node_info(element, ctx.nodestats, no_insert=True)
    nd_stat = node_info.stats
    node_info.add_html_info['dbce-class'] = node_info.dbce_class
    node_info.add_html_info['dbce-wc'] = nd_stat['word_cnt']
    node_info.add_html_info['dbce-tc'] = nd_stat['tag_cnt']
    node_info.add_html_info['dbce-hrefs'] = nd_stat['href_cnt']
    node_info.add_html_info['dbce-w2t'] = node_info.w2t_ratio
    node_info.add_html_info['dbce-t2w'] = node_info.t2w_ratio
    node_info.add_html_info['dbce-lden'] = node_info.link_density
    node_info.add_html_info['dbce-tden'] = node_info.text_density
    node_info.add_html_info['dbce-id'] = node_info.nodeid
    if node_info.preprocess_class:
        node_info.add_html_info['dbce-pp-class'] = element.preprocess_class

def preprocess_fix_child(element, ctx):
    el_class = get_node_info(element, ctx.nodestats, no_insert=True).dbce_class
    for child in list(filter_tag(element.children)):
        if get_node_info(child, ctx.nodestats, no_insert=True) is None:
            set_nodes_class(child, el_class, ctx)

def stage2_step_classify(elements, ctx, testname, versions=None):
    """This function iterates over all elements of one group (same parent)
    and classifies <dbce> nodes based on the bit patterns.
    """
    _elements = list(filter_tag(elements))
    invalid_as_bp = True if testname in ['T002'] else False
    for element in _elements:
        if is_tag(element, [DBCE_MARKUP_ANALYSIS_TAG]):
            _class = classify_block(versions, marker_to_bits(element['class']),)
            if invalid_as_bp and _class == DBCE_CLASS_INVALID:
                _class = DBCE_CLASS_BOILER_PLATE
            el_info = get_node_info(element, ctx.nodestats)
            get_xpath(element, ctx)
            set_nodes_class(element, _class, ctx)
            if el_info.preprocess_class is None:
                el_info.preprocess_class = _class
            else:
                raise ValueError
            preprocess_fix_child(element, ctx)

def node_stats(node, ctx):
    node_info = get_node_info(node, ctx.nodestats, no_insert=True)
    if node_info.stats_done:
        return

    el_stats = node_info.stats

    for child in node.children:
        if is_string(child):
            for token in child.string.split():
                el_stats['word_cnt'] += 1
                el_stats['text_len'] += len(token.strip()) + 1
            el_stats['num_wraped_lines'] = round(el_stats['text_len']/80) \
                                           + (1 if  el_stats['text_len']%80 else 0)
        elif is_tag(child, any_tag=True):
            if child.name not in [DBCE_MARKUP_ANALYSIS_TAG, 'br']:
                el_stats['tag_cnt'] += 1
            if child.name == 'a':
                el_stats['href_cnt'] += 1
                el_stats['text_in_links'] += len(child.text.strip().split())

        if isinstance(child, bs4.element.Tag):
            child_info = get_node_info(child, ctx.nodestats, no_insert=True)
            node_info.sum_stats(child_info)

    node_info.w2t_ratio = ratio(el_stats['word_cnt'], el_stats['tag_cnt'])
    node_info.t2w_ratio = ratio(el_stats['tag_cnt'], el_stats['word_cnt'])
    node_info.link_density = ratio(el_stats['text_in_links'], el_stats['word_cnt'])
    node_info.text_density = ratio(el_stats['word_cnt'], el_stats['num_wraped_lines'])
    node_info.stats_done = True

def set_nodes_class(node, cls, ctx, force=False):
    node_info = get_node_info(node, ctx.nodestats, no_insert=True)

    if node_info.dbce_class is not None and node_info.dbce_class != DBCE_CLASS_UNKNWON:
        logger.debug("# you want me to change <{}> class? ({}) {} to {}".format(node_info.node_name,
                                                                                node_info.nodeid,
                                                                                node_info.dbce_class,
                                                                                cls))
        if node_info.dbce_class != cls and not force:
            raise ValueError("class change {} -> {} not allowed".format(node_info.dbce_class, cls))
    else:
        logger.debug("# you want me to set <{}> class? ({}) {} to {}".format(node_info.node_name,
                                                                             node_info.nodeid,
                                                                             node_info.dbce_class,
                                                                             cls))
    node_info.dbce_class = cls
    node_stats(node, ctx)

def get_class_stats(elements, ctx):
    _classes = list([get_node_info(_, ctx.nodestats, no_insert=True).dbce_class for _ in elements])
    return (_classes, Counter(_classes))

def iterate_over_elements(elements, ctx):
    nodes_iter = iter(elements)
    dummy_node = get_dummy_node(ctx)
    prev_node = cur_node = next_node = dummy_node

    try:
        cur_node = next(nodes_iter)
    except StopIteration:
        pass
    try:
        next_node = next(nodes_iter)
    except StopIteration:
        pass

    yield (get_node_info(prev_node, ctx.nodestats, no_insert=True),
           get_node_info(cur_node, ctx.nodestats, no_insert=True),
           get_node_info(next_node, ctx.nodestats, no_insert=True))

    if next_node != dummy_node:
        stop = False
        while not stop:
            prev_node = cur_node
            cur_node = next_node
            try:
                next_node = next(nodes_iter)
            except StopIteration:
                next_node = dummy_node
                stop = True
            yield (get_node_info(prev_node, ctx.nodestats, no_insert=True),
                   get_node_info(cur_node, ctx.nodestats, no_insert=True),
                   get_node_info(next_node, ctx.nodestats, no_insert=True))

def stage2_step_classify_postprocess(elements, ctx):
    _elements = list(filter_tag(elements))
    _elements.reverse()
    _elements_count = len(_elements)

    if _elements_count < 1:
        return

    for element in _elements:
        el_info = get_node_info(element, ctx.nodestats, no_insert=True)
        if el_info.dbce_class is None:
            el_info.dbce_class = DBCE_CLASS_UNKNWON
            node_stats(element, ctx)

    (_classes, _classes_cnt) = get_class_stats(_elements, ctx)
    percent_by_class = compute_class_distribution(_elements, ctx)

    logger.debug("# stage #1: ids  ({}): {}".format(_elements_count,
                                                    [(_.name, id(_)) for _ in _elements]))
    logger.debug("# stage #1: classes: {} / {}".format(_classes, _classes_cnt))
    logger.debug("# stage #1: distribution: {}".format(percent_by_class))

    if _elements_count == 1 or is_list_equal(_classes):
        _el0_info = get_node_info(_elements[0], ctx.nodestats, no_insert=True)
        if _el0_info.dbce_class != DBCE_CLASS_UNKNWON:
            set_nodes_class(_el0_info.parent, _el0_info.dbce_class, ctx)
        node_stats(_elements[0], ctx)
    else:
        logger.warning("# stage #1: ec: {} and classes not equal".format(_elements_count))

def tag_to_id(tag):
    try:
        return DBCE_VALID_TAGS.index(tag)
    except ValueError:
        return -1

def node_class_to_id(node):
    if 'class' in node.attrs:
        return hash("-".join(node.attrs['class']))
    else:
        return -1

def get_parent_environ(ctx, parent):
    if not parent:
        return None

    if parent.node_name in ['a']:
        return get_parent_environ(ctx, get_node_info(parent.parent, ctx.nodestats, no_insert=True))

    prev_sib = None
    next_sib = None

    for el in filter_tag(parent.node.previousSiblingGenerator()):
        prev_sib = get_node_info(el, ctx.nodestats, no_insert=True)
        break

    for el in filter_tag(parent.node.nextSiblingGenerator()):
        next_sib = get_node_info(el, ctx.nodestats, no_insert=True)
        break

    return (prev_sib, next_sib)

def get_direct_vector(ctx, prev_node, cur_node, next_node):
    dummy_node = get_node_info(get_dummy_node(ctx), ctx.nodestats, no_insert=True)

    if not prev_node:
        prev_node = dummy_node
    if not next_node:
        next_node = dummy_node
    _known_tags = sorted(list(ctx.seen_tags))
    _node_class = -1

    if cur_node.node_name != 'dbce':
         _cls = cur_node.node.attrs.get('class', [])
         if len(_cls):
             _cls.sort()
             _node_class = hash(" ".join(_cls))

    vector = (
        _known_tags.index(prev_node.node_name),
        _known_tags.index(cur_node.node_name),
        _known_tags.index(next_node.node_name),
        _node_class,
        prev_node.link_density,
        prev_node.text_density,
        cur_node.link_density,
        cur_node.text_density,
        next_node.link_density,
        next_node.text_density,
        prev_node.stats['word_cnt'],
        cur_node.stats['word_cnt'],
        next_node.stats['word_cnt'],
        prev_node.stats['text_len'],
        cur_node.stats['text_len'],
        next_node.stats['text_len'],
        prev_node.stats['tag_cnt'],
        cur_node.stats['tag_cnt'],
        next_node.stats['tag_cnt'],
        ratio(cur_node.stats['text_len'], cur_node.stats['tag_cnt'])
    )
    return vector

def get_vector(ctx, prev_node, cur_node, next_node, parent=None, testname=None):

    (p_prev, p_next) = get_parent_environ(ctx, parent)
    level0_vect = get_direct_vector(ctx, prev_node, cur_node, next_node)

    _ret_val = level0_vect

    if ctx.testname in [None, 'T004', 'T006']:
        level1_vect = get_direct_vector(ctx, p_prev, parent, p_next)
        _ret_val += level1_vect
        if ctx.testname not in [None, 'T005', 'T006']:
            return _ret_val

    if ctx.testname in [None, 'T005', 'T006']:
        (p_prev, p_next) = get_parent_environ(ctx,
                                              get_node_info(parent.parent,
                                                            ctx.nodestats, no_insert=True))
        level2_vect = get_direct_vector(ctx, p_prev, parent, p_next)
        _ret_val += level2_vect
    return _ret_val

def classify_elements_stage3(elements, ctx, clf=None, testname=None):
    _classes = []

    # all nodes share the same parent
    if len(elements) < 1:
        logger.warning("not enough elements...")
        return

    parent = get_node_info(elements[0].parent, ctx.nodestats, no_insert=True)
    _vectors = []
    for (prev_node, cur_node, next_node) in iterate_over_elements(elements, ctx):
        # logger.debug("# PARENT dbce link/text density: {} ({})/ {}".format(
        #     id(parent.node),
        #     parent.node_name,
        #     (parent.link_density, parent.text_density)))

        # logger.debug("# nodes       : {} / {} ({}) / {}".format(prev_node.node_name,
        #                                                         cur_node.node_name,
        #                                                         id(cur_node),
        #                                                         next_node.node_name))
        # logger.debug("# dbce classes: {} / {} / {}".format(prev_node.dbce_class,
        #                                                    cur_node.dbce_class,
        #                                                    next_node.dbce_class))
        # logger.debug("# dbce link/text density: {} / {} / {}".format(
        #     (prev_node.link_density, prev_node.text_density),
        #     (cur_node.link_density, cur_node.text_density),
        #     (next_node.link_density, next_node.text_density)))

        # logger.debug("# " + "-"*20)

        _vec = get_vector(ctx, prev_node, cur_node, next_node, parent, testname=testname)

        # logging even preparation of the log message is to expensive
        if logger.level == logging.DEBUG:
            logger.debug(" vector : {}".format(_vec))
        _vectors.append(_vec)

    _oracle = clf.predict(_vectors)
    if logger.level == logging.DEBUG:
        logger.debug(" oracle: {}".format(_oracle))
    def r2class(result):
        if result == 1:
            return DBCE_CLASS_CONTENT
        else:
            return DBCE_CLASS_BOILER_PLATE

    re_classes = map(r2class, _oracle)
    return re_classes

def get_dummy_node(ctx):
    get_node_info(DUMMY_NODE, ctx.nodestats)
    return DUMMY_NODE

def stage2_step_reclassify(elements, ctx, clf=None, testname=None):
    _elements = list(filter_tag(elements))
    _elements.reverse()

    if len(_elements) < 1:
        return

    (_classes, _classes_cnt) = get_class_stats(_elements, ctx)
    _all_class_eq = is_list_equal(_classes)
    _el0_info = get_node_info(_elements[0], ctx.nodestats, no_insert=True)
    _el0_parent_info = get_node_info(_el0_info.parent, ctx.nodestats, no_insert=True)

    # FIXME: check if this is needed ... was copy&pasted from stage2
    #if _all_class_eq and _el0_parent_info.dbce_class == _classes[0] \
    #   and _classes[0] != DBCE_CLASS_UNKNWON:
    #    return

    # we have the same parent for all elements
    parent = _el0_parent_info.node
    logger.debug("# stage #3: classes: {} / {} / {}".format(_classes, _classes_cnt, _all_class_eq))
    # logger.debug("# stage #3: distribution: {}".format(percent_by_class))

    tmp_classes_lr = classify_elements_stage3(_elements, ctx, clf, testname=testname)

    changes = False
    for (node, cls) in zip(_elements, tmp_classes_lr):
        node_info = get_node_info(node, ctx.nodestats, no_insert=True)
        if cls is None:
            continue
        elif cls != node_info.dbce_class:
            # FIXME: define a testcase for this one or remove it
            # jusTex method
            logger.debug("# stage #3: reclassify id: {}".format(node_info.nodeid))
            if cls == DBCE_CLASS_BOILER_PLATE and node_info.parent.name in ['h1', 'h2', 'h3', 'h4']:
                logger.error("take a look on this one - node_name is hX")
                #set_nodes_class(node, DBCE_CLASS_UNKNWON, ctx, force=True)
            else:
                # FIXME: make the change based on score?!
                set_nodes_class(node, cls, ctx, force=True)
            changes = True
        else:
            pass

    if not changes:
        return

    (_classes, _classes_cnt) = get_class_stats(_elements, ctx)
    _all_class_eq = is_list_equal(_classes)
    logger.debug("# stage #3: RE classes: {} / {} / {}".format(_classes, _classes_cnt, _all_class_eq))
    if _all_class_eq:
        logger.debug("# stage #3: RE all classes are equal... set up parent")
        set_nodes_class(parent, get_node_info(_elements[0], ctx.nodestats, no_insert=True).dbce_class,
                        ctx, force=True)
    else:
        # FIXME: this should be extended...
        logger.debug("# stage #3: just skip it")

def compute_class_distribution(_elements, ctx):
    (_classes, class_cnt) = get_class_stats(_elements, ctx)

    all_classes_cnt = class_cnt[DBCE_CLASS_CONTENT] \
                      + class_cnt[DBCE_CLASS_BOILER_PLATE] \
                      + class_cnt[DBCE_CLASS_INVALID] \
                      + class_cnt[DBCE_CLASS_UNKNWON]
    percent_bp = percentage(class_cnt[DBCE_CLASS_BOILER_PLATE], all_classes_cnt)
    percent_con = percentage(class_cnt[DBCE_CLASS_CONTENT], all_classes_cnt)
    percent_inv = percentage(class_cnt[DBCE_CLASS_INVALID], all_classes_cnt)
    percent_unk = percentage(class_cnt[DBCE_CLASS_UNKNWON], all_classes_cnt)

    percent_by_class = {DBCE_CLASS_CONTENT: percent_con,
                        DBCE_CLASS_BOILER_PLATE: percent_bp,
                        DBCE_CLASS_INVALID: percent_inv,
                        DBCE_CLASS_UNKNWON: percent_unk,
                        None: 0}
    return percent_by_class

def dbce_method(ctx, soup, versions=None, testname=None):
    """ Retruns BeautifulSoup object with merged and reclassified <dbce> blocks"""
    soup_body = soup.body

    # timing
    ctx.time_s2_s_classify_start = time.time()

    layers = get_layers(soup_body)
    for level in sorted(layers.keys(), reverse=True):
        for nodes in filter(lambda x: x, layers[level]):
            # 0) classify all <dbce>
            stage2_step_classify(nodes, ctx=ctx, testname=testname, versions=versions)

    # timing
    ctx.time_s2_s_classify_end = time.time()
    ctx.time_s2_s_classify_pp_start = time.time()

    for level in sorted(layers.keys(), reverse=True):
        for nodes in layers[level]:
            # FIXME: sanity check remove it later
            # if None in list(map(lambda x: get_node_info(x, ctx.nodestats, no_insert=True).dbce_class,
            #                     filter_tag(nodes, check=[DBCE_MARKUP_ANALYSIS_TAG]))):
            #    raise ValueError("dbce elements without class - how?!")
            # 1) if alone forward up
            stage2_step_classify_postprocess(nodes, ctx)
    ctx.time_s2_s_classify_pp_end = time.time()
    ctx.time_s2_s_learn_start = time.time()
    # learn classifier
    if testname in [None, 'T003', 'T004', 'T005', 'T006']:
        (clf, vectors) = stage2_step_learn(ctx, soup_body)
    ctx.time_s2_s_learn_end = time.time()

    ctx.time_s2_s_reclassify_start = time.time()
    if testname in [None, 'T003', 'T004', 'T005', 'T006']:
        for level in sorted(layers.keys()):
            for nodes in layers[level]:
                stage2_step_reclassify(nodes, ctx, clf, testname=testname)
        ctx.clf_clf = clf
        ctx.clf_vectors = vectors
    ctx.time_s2_s_reclassify_end = time.time()

def extract_content_without_dbce(soup, ctx):
    out = []
    for node in filter_tag(soup.body.recursiveChildGenerator()):
        if is_content(get_node_info(node, ctx.nodestats, no_insert=True)):
            if node.string is not None:
                for _ in textwrap.wrap(node.string.strip(), 70):
                    out.append(_)

    return "\n".join(out)

def extract_content(soup, ctx):
    out = []
    for node in filter_tag(soup.body.recursiveChildGenerator(), ['dbce']):
        if is_content(get_node_info(node, ctx.nodestats, no_insert=True)):
            for _ in textwrap.wrap(node.text.strip(), 70):
                out.append(_)

    return "\n".join(out)

def collect_vectors(ctx, soup):
    vectors = OrderedDict()

    layers = get_layers(soup)
    for level in sorted(layers.keys(), reverse=True):
        for nodes in layers[level]:
            _elements = list(filter_tag(nodes))
            if len(_elements) == 0:
                continue

            # all elements have the same parent so let's get it only once
            # bs4 is awful slow __getattr__.find() ...
            parent = get_node_info(_elements[0].parent, ctx.nodestats, no_insert=True)
            get_xpath(parent.node, ctx)
            for (prev_node, cur_node, next_node) in iterate_over_elements(_elements, ctx):

                get_xpath(cur_node.node, ctx)
                if not (prev_node.dbce_class in [DBCE_CLASS_CONTENT, DBCE_CLASS_BOILER_PLATE] \
                        or cur_node.dbce_class in [DBCE_CLASS_CONTENT, DBCE_CLASS_BOILER_PLATE] \
                        or next_node.dbce_class in [DBCE_CLASS_CONTENT, DBCE_CLASS_BOILER_PLATE]):
                    continue

                cur_node.clf_vector = get_vector(ctx, prev_node, cur_node, next_node, parent)
                _ = vectors.get(cur_node.clf_vector, False)
                if is_content(cur_node):
                    if _ and _ != 1:
                        logger.error("vector conflict bp: {}".format(cur_node.clf_vector))
                        get_parent_environ(ctx, parent)
                        del vectors[cur_node.clf_vector]
                        continue
                    vectors[cur_node.clf_vector] = (cur_node.xpath, 1)
                else:
                    if _ and _ != 0:
                        logger.error("vector conflict content: {}".format(cur_node.clf_vector))
                        get_parent_environ(ctx, parent)
                        del vectors[cur_node.clf_vector]
                        continue
                    vectors[cur_node.clf_vector] = (cur_node.xpath, 0)

    return vectors

def stage2_step_learn(ctx, soup):
    dataset = []
    datacls = []
    vectors = collect_vectors(ctx, soup)
    for (vec, cls) in vectors.items():
        dataset.append(vec)
        datacls.append(cls[1])

    logger.debug("## using {} vectors ".format(len(datacls)))

    if not ctx.testname:
        raise RuntimeError

    if ctx.testname in ['T006']:
        clf = RandomForestClassifier()
        ctx.clf_name = "RandomForestClassifier"

    if ctx.testname in ['T003', 'T004', 'T005']:
        clf = KNeighborsClassifier()
        ctx.clf_name = "KNeighborsClassifier"

    clf.fit(dataset, datacls)
    # prepare data for hyper parameter tunning
    _ = ctx.clf_data.setdefault('clf_agg_dataset', [])
    _.extend(dataset)
    _ = ctx.clf_data.setdefault('clf_agg_datacls', [])
    _.extend(datacls)

    return (clf, vectors)

class DBCEMethodLSH:
    """ Class for automated analysis using LSH and available
    informations from journal.

    Journal contains all data extracted from WARC files.
    """

    def __init__(self, config, cross, testname=None):
        """ Create DBCEMethodLSH object.

        cross:        container with chosen entries (cur, up, down, prev, next)
        parser:       HTML parser accepted by BeautifulSoup
        logger:       (optional) logger
        """

        self.cross = cross
        self.testname = testname
        self.domain = cross.cur['url_parsed'].netloc
        self.parser = config.get(self.domain, 'parser')

    def setup_dataset(self):
        _data_set = []
        if self.cross.simup:
            _data_set.append((self.cross.simup['bs_html_sanitized'].body.prettify(),
                              DBCE_DIFF_UP))

        if self.cross.simdown:
            _data_set.append((self.cross.simdown['bs_html_sanitized'].body.prettify(),
                              DBCE_DIFF_DOWN))

        if self.cross.prev:
            _data_set.append((self.cross.prev['bs_html_sanitized'].body.prettify(),
                              DBCE_DIFF_PREV))

        if self.cross.next:
            _data_set.append((self.cross.next['bs_html_sanitized'].body.prettify(),
                              DBCE_DIFF_NEXT))

        _data_set.append((self.cross.cur['bs_html_sanitized'].body.prettify(), DBCE_DIFF_CUR))

        logger.info("using {} sources to construct cross".format(len(_data_set)))
        return _data_set

    def stage2_step_annotate(self, ctx, data_set, dbce_markup=dbce_markup):
        ctx.time_s2_s_annotate_start = time.time()

        opt_compress = ctx.config.getboolean('DEFAULT', 'compress_blocks')

        result = html_annotate(data_set, markup=dbce_markup,
                               compress=opt_compress, window_size=self.cross.window_size)

        ctx.time_s2_s_annotate_end = time.time()

        _hdata = BeautifulSoup('<html><head>{}</head>{}</html>'.format(
            self.cross.cur['bs_html_sanitized'].head.prettify(),
            result),
                               'lxml')
        return (result, _hdata)

    def analysis(self, ctx):
        """ Returns an analysis object (dict). This object contains
        all informations and results needed to dump the data.
        """
        ctx.time_analysis_start = time.time()

        _data_set = self.setup_dataset()
        ######
        # dump annotated HTML document
        # this is not part of the DBCE-Method
        _hdata = self.stage2_step_annotate(ctx, _data_set, dbce_markup)[1]
        ctx.bs_html_sanitized = self.cross.cur['bs_html_sanitized']

        cur_url_parsed = urlparse(self.cross.cur['url'])
        inject_header(_hdata, cur_url_parsed)
        versions = self.cross.to_bits()

        _hdata_backup = copy.copy(_hdata)
        _init_stats(ctx, _hdata.body)

        wrap_with_div(_hdata, ctx, versions, clean_markup=False)
        ctx.bs_diff = BeautifulSoup(_hdata.prettify(), 'lxml')
        # end of dump
        ######

        ######
        # actuall DBCE-Method
        #
        # reset node's stats because we work now on a copy
        ctx.nodestats = NodeStats()
        _init_stats(ctx, _hdata_backup)

        ctx.time_dbce_method_start = time.time()
        dbce_method(ctx, _hdata_backup, versions=versions, testname=self.testname)
        ctx.time_dbce_method_end = time.time()
        ctx.time_analysis_end = time.time()

        dump_context_info(ctx, self.cross, "reclass")
        dump_nodes_info(ctx, ctx.nodestats, "reclass")

        ctx.extracted_data = extract_content(_hdata_backup, ctx)
        # dump final HTML document
        wrap_with_div(_hdata_backup, ctx, versions, clean_markup=True, use_dbce=False)
        ctx.bs_diff_merged = BeautifulSoup(_hdata_backup.prettify(), 'lxml')
