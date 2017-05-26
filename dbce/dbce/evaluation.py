"""
evaluation - Set of functions for result evaluation.
Copyright (c) 2016 Rafal Lesniak

This software is licensed as described in the file LICENSE.
"""

import os
import pdb
import re

from collections import Counter

import itertools
import numpy as np
import matplotlib as mpl
# disable X11
#mpl.use('pgf')

import matplotlib.pyplot as plt

import pandas as pd

from sklearn.externals import joblib
from sklearn.metrics import (confusion_matrix, classification_report, f1_score,
                             precision_score, recall_score, roc_auc_score,
                             make_scorer, accuracy_score)
from sklearn.model_selection import train_test_split
from sklearn.model_selection import (GridSearchCV, RandomizedSearchCV)
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

from dbce.constants import (DBCE_CLASS_CONTENT, DBCE_CLASS_BOILER_PLATE)
from dbce.datatypes.evaluation import (EvalResult, EvalResultContainer)
from dbce.utils import load_groundtruth_for_cross

RE_DF_NODES = re.compile(r'.*df\-reclass\-(?P<docid>.*)\-nodes\.csv$')
RE_DF_NODE_VEC = re.compile(r'.*df\-reclass\-(?P<docid>.*)\-node\-vec\.csv$')
RE_DF_NODE_CTX = re.compile(r'.*df\-reclass\-(?P<docid>.*)\-node\-ctx\.csv$')

def _parse_report(report):
    (p_bp, r_bp, f1_bp, sup_bp) = list(filter(lambda x: x != '',
                                              report.splitlines()[2].split(" ")))[1:]
    (p_con, r_con, f1_con, sup_con) = list(filter(lambda x: x != '',
                                                  report.splitlines()[3].split(" ")))[1:]
    (p_avg, r_avg, f1_avg, sub_avg) = list(filter(lambda x: x != '',
                                                  report.splitlines()[5].split(" ")))[3:]
    ret = (float(p_bp), float(r_bp), float(f1_bp), int(sup_bp),
           float(p_con), float(r_con), float(f1_con), int(sup_con),
           float(p_avg), float(r_avg), float(f1_avg), int(sub_avg))
    print(report)
    print(ret)
    print("#"*30)
    return ret

def tune_hyper_parameter(config, crosses, data, domain, workdir):
    _dataset = data['agg_dataset']
    _datacls = data['agg_datacls']

    groundtruth = {}
    for bits in crosses:
        for cross in crosses[bits]:
            for (gt_hash, gt_xpath, gt_cls) in load_groundtruth_for_cross(
                    config, cross[3]):
                groundtruth[gt_hash] = gt_cls

    dataset = []
    datacls = []
    gt_set = []
    gt_cls = []

    for (cls_hash, cls_vec, cls_class) in data['clf_vectors']:
        if cls_hash in groundtruth:
            _dataset.append(cls_vec)
            _datacls.append(cls_class)
            gt_set.append(cls_vec)
            gt_cls.append(groundtruth[cls_hash])

    x_train = np.array(_dataset)
    x_test = np.array(gt_set)
    y_train = np.array(_datacls)
    y_test = np.array(gt_cls)

    results = []

    for _scorer in [make_scorer(accuracy_score), make_scorer(f1_score),
                    make_scorer(precision_score), make_scorer(recall_score)]:
        for (label, model, param_grid) in get_model_candidates():
            clf = RandomizedSearchCV(model,
                                     param_grid,
                                     #cv=5,
                                     scoring=_scorer,
                                     verbose=0,
                                     n_jobs=-1,
                                     random_state=0)
            # clf = GridSearchCV(model,
            #                    param_grid,
            #                    #cv=5,
            #                    scoring=_scorer,
            #                    verbose=1,
            #                    n_jobs=-1)


            clf.fit(x_train, y_train)
            print("Best parameters set found on development set: {} {}".format(label, clf.best_params_))
            print("Best estimator: {}".format(clf.best_estimator_))
            print("Best score: {}".format(clf.best_score_))
            dump_gridsearch_result(clf, label, _scorer, domain, workdir)
    return clf

def get_model_candidates():
    candidates = []
    svm_tuned_param = {'kernel': ['poly', 'rbf', 'linear'],
                       'max_iter': range(0, 500),
                       'degree': range(1, 10)}
    candidates.append(["SVM", SVC(C=1), svm_tuned_param])

    _max_depth = [None]
    _max_depth.extend(range(1, 200))
    rf_tuned_param = {'n_estimators': range(2, 100),
                      'min_samples_split': range(2, 200),
                      'max_depth': _max_depth,
                      'min_samples_leaf': range(1, 200),
                      'criterion': ['gini', 'entropy'],
                      'class_weight': [None, 'balanced']}
    candidates.append(["RandomForest",
                       RandomForestClassifier(n_jobs=-1),
                       rf_tuned_param])

    knn_tuned_param = {
        'n_neighbors': range(1, 100),
        'leaf_size': range(1, 1000),
    }
    candidates.append(["kNN", KNeighborsClassifier(),
                       knn_tuned_param])
    return candidates

def dump_eval_result(ctx, result):
    dumpfile = os.path.join(ctx.output_dir, "eval-result-{}.pkl".format(result.docid))
    joblib.dump(result, dumpfile)

def dump_gridsearch_result(clf, label, scorer, domain, output_dir):
    dumpfile = os.path.join(output_dir, "rsCB-result-{}-{}.txt".format(label, domain))
    with open(dumpfile, 'a') as fout:
        fout.write(" # scorer {}\n".format(scorer))
        fout.write(" # best params\n")
        fout.write(" -"*10)
        fout.write("\n")
        fout.write(str(clf.best_params_))
        fout.write("\n")
        fout.write(repr(clf.best_estimator_))
        fout.write("\n")
        means = clf.cv_results_['mean_test_score']
        stds = clf.cv_results_['std_test_score']
        for mean, std, params in zip(means, stds, clf.cv_results_['params']):
            fout.write("%0.3f (+/-%0.03f) for %r\n"
                       % (mean, std * 2, params))

def _collect_accuracy(config, agg_ctx, agg_nodes, report_dir):

    agg_acc_df = pd.DataFrame(columns=['domain', 'docid', 'testname', 'dbce_nodes', 'dbce_gt_nodes',
                                       'cross_unique_id',
                                       'acc', 'tp', 'fn', 'fp', 'tn', 'p_bp', 'r_bp', 'f1_bp',
                                       'sup_bp', 'p_con', 'r_con', 'f1_con', 'sup_con',
                                       'p_avg', 'r_avg', 'f1_avg', 'sub_avg'])
    gt_classes = {}
    clf_classes = {}
    pre_classes = {}
    class_labels = [DBCE_CLASS_BOILER_PLATE, DBCE_CLASS_CONTENT]

    _done_bits = {}

    testnames = agg_ctx.groupby('testname')

    for testname in testnames.groups.keys():
        _my_group = testnames.get_group(testname)
        for (idx, docid, domain, _c_u_id) in _my_group[['docid', 'domain', 'cross_unique_id']].to_records():
            _docid = docid
            groundtruth = gt2dict(config, _c_u_id)

            _ = gt_classes.setdefault(domain, {})
            _.setdefault(testname, [])
            _ = clf_classes.setdefault(domain, {})
            _.setdefault(testname, [])
            _ = pre_classes.setdefault(domain, {})
            _.setdefault(testname, [])

            xph2cls = {}
            prexph2cls = {}
            _dbce_nodes = 0
            _dbce_gt_nodes = 0
            for _ in agg_nodes.loc[(agg_nodes['docid'] == docid) & (agg_nodes['testname'] == testname)].to_records():

                if _[6] != 'dbce':
                    continue

                if pd.isnull(_[11]):
                    continue

                _dbce_nodes += 1

                if _[8] in ['dbce-class-invalid', 'dbce-class-unk']:
                    continue

                if _[7] in ['dbce-class-invalid', 'dbce-class-unk']:
                    continue

                _hash = hash(_[11])
                if _hash in xph2cls and xph2cls[_hash] != _[11]:
                    raise ValueError
                xph2cls[_hash] = _[8]

                if _hash in prexph2cls and prexph2cls[_hash] != _[7]:
                    raise ValueError

                prexph2cls[_hash] = _[7]
            _gt_classes = []
            _clf_classes = []
            _pre_classes = []

            for xpathhash in groundtruth:
                xph = int(xpathhash)
                if xph not in xph2cls:
                    continue
                else:
                    _dbce_gt_nodes += 1
                    _gt_classes.append(groundtruth[xph])
                    _clf_classes.append(xph2cls[xph])
                    _pre_classes.append(prexph2cls[xph])

            c_matrix = confusion_matrix(_gt_classes, _clf_classes,
                                        [DBCE_CLASS_BOILER_PLATE, DBCE_CLASS_CONTENT])
            print("{} {} total classes: {}".format(domain, testname, len(_gt_classes)))
            print(c_matrix)
            _report = classification_report(_gt_classes, _clf_classes, labels=[DBCE_CLASS_BOILER_PLATE, DBCE_CLASS_CONTENT])
            (p_bp, r_bp, f1_bp, sup_bp,
             p_con, r_con, f1_con, sup_con,
             p_avg, r_avg, f1_avg, sup_avg) = _parse_report(_report)
            # http://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html
            agg_acc_df = agg_acc_df.append(pd.DataFrame([[domain,
                                                          _docid,
                                                          testname,
                                                          _dbce_nodes,
                                                          _dbce_gt_nodes,
                                                          _c_u_id,
                                                          accuracy_score(_gt_classes,
                                                                         _clf_classes),
                                                          c_matrix[1][1], # tp
                                                          c_matrix[1][0], # fn
                                                          c_matrix[0][1], # fp
                                                          c_matrix[0][0], # tn
                                                          p_bp, r_bp, f1_bp, sup_bp,
                                                          p_con, r_con, f1_con, sup_con,
                                                          p_avg, r_avg, f1_avg, sup_avg]],
                                                        columns=agg_acc_df.columns))

            gt_classes[domain][testname].extend(_gt_classes)
            clf_classes[domain][testname].extend(_clf_classes)
            pre_classes[domain][testname].extend(_pre_classes)

    return agg_acc_df

def load_eval_results(config, report_dir):
    agg_ctx = []
    agg_vec = []
    agg_nodes = []

    filenames = []
    for _ in os.listdir(report_dir):
        filename = os.path.join(report_dir, _)
        filenames.append(filename)

    agg_ctx = pd.read_csv(os.path.join(report_dir, 'df-ctx.csv'),
                          verbose=True,
                          error_bad_lines=True)

    agg_nodes = pd.read_csv(os.path.join(report_dir, 'df-nodes.csv'),
                            verbose=True, error_bad_lines=True)
    agg_nodes = agg_nodes.reset_index(drop=True)

    agg_vec = pd.read_csv(os.path.join(report_dir, 'df-node-vec.csv'),
                          verbose=True, error_bad_lines=False)

    for rfile in map(RE_DF_NODES.match, filenames):
        if not rfile:
            continue

        dataframe = pd.read_csv(rfile.string, verbose=True, error_bad_lines=False)
        dataframe['docid'] = rfile.groupdict()['docid']
        agg_nodes.append(dataframe)

    agg_ctx['s2_s_classify'] \
        = agg_ctx['time_s2_s_classify_end']-agg_ctx['time_s2_s_classify_start']

    agg_ctx['s2_s_annotate'] \
        = agg_ctx['time_s2_s_annotate_end']-agg_ctx['time_s2_s_annotate_start']

    agg_ctx['s2_s_learn'] \
        = agg_ctx['time_s2_s_learn_end']-agg_ctx['time_s2_s_learn_start']

    agg_ctx['s2_s_classify_pp'] \
        = agg_ctx['time_s2_s_classify_pp_end']-agg_ctx['time_s2_s_classify_pp_start']

    agg_ctx['s2_s_reclassify'] \
        = agg_ctx['time_s2_s_reclassify_end']-agg_ctx['time_s2_s_reclassify_start']

    agg_ctx['dbce_total'] \
        = agg_ctx['time_dbce_method_end']-agg_ctx['time_dbce_method_start'] \
                               + agg_ctx['s2_s_annotate']

    agg_ctx['analysis_total'] \
        = agg_ctx['time_analysis_end']-agg_ctx['time_analysis_start']

    agg_ctx['gt_bp_cnt'] = 0
    agg_ctx['gt_con_cnt'] = 0

    grouped = agg_ctx.groupby('domain')
    for domain in grouped.groups.keys():
        plt.figure()
        ctx_grp = grouped.get_group(domain)
        rara = ctx_grp[['s2_s_annotate', 's2_s_classify', 's2_s_classify_pp',
                        's2_s_learn',
                        's2_s_reclassify']]

        rara = rara.rename_axis({"s2_s_annotate": "annotate", 's2_s_classify': "classify",
                                 's2_s_classify_pp': "classify_pp", 's2_s_learn': "learn",
                                 "s2_s_reclassify": 'reclassify'}, axis="columns")
        rara.plot.box(fontsize=16)
        _fname = os.path.join(report_dir, '{}-agg-timing'.format(domain))
        ax = plt.axes()
        ax.set_ylabel('time in seconds')
        cf = plt.gcf()
        _title = cf.suptitle('{}'.format(domain))
        _title.set_fontsize(20)

        plt.savefig(_fname + '.pdf')
        plt.close('all')

    plt.figure()
    agg_ctx.boxplot(column=['dbce_total'], by='domain', rot=45, fontsize=16)
    ax = plt.axes()
    ax.set_ylabel('time in seconds')
    ax.set_title('')
    cf = plt.gcf()
    cf.suptitle('')
    plt.tight_layout()

    _fname = os.path.join(report_dir, 'all-agg-timing')
    plt.savefig(_fname + '.pdf')
    plt.close('all')

    agg_nodes.groupby('domain')[['nodeid']].count().plot.bar(rot=45, label='DOM Elements')

    ax = plt.axes()
    ax.set_ylabel('Number of DOM elements')
    ax.set_title('')
    cf = plt.gcf()
    cf.suptitle('')
    plt.tight_layout()

    _fname = os.path.join(report_dir, 'all-agg-nodes')
    plt.savefig(_fname + '.pdf')
    plt.close('all')

    # compute accuracy for each domain
    agg_acc_df = _collect_accuracy(config, agg_ctx, agg_nodes, report_dir)

    plt.close('all')
    plt.figure()
    agg_acc_df.boxplot(column='acc', by=['domain', 'testname'], rot=90)
    ax = plt.axes()
    ax.set_ylabel('accuracy [%]')
    ax.set_title('')
    cf = plt.gcf()
    cf.suptitle('')
    plt.tight_layout()
    _fname = os.path.join(report_dir, 'all-box-acc')
    plt.savefig(_fname + '.pdf')

    agg_acc_df.boxplot(column='acc', by=['testname'], fontsize=16)
    ax = plt.axes()
    ax.set_ylabel('accuracy [%]')
    ax.set_title('')
    cf = plt.gcf()
    cf.suptitle('')
    plt.tight_layout()

    _fname = os.path.join(report_dir, 'all-box-acc-test')
    plt.savefig(_fname + '.pdf')
    plt.close('all')

    prf1_df = agg_acc_df[['domain', 'docid', 'testname', 'tp', 'fn', 'fp', 'tn', 'acc',
                          'p_bp', 'p_con', 'r_bp',
                          'r_con', 'f1_bp', 'f1_con']]

    grp = prf1_df.groupby('testname')

    for testname in ['T001', 'T002', 'T006']:
        grp1 = grp.get_group(testname)
        _raw_data = grp1[['tp', 'fn', 'fp', 'tn']].copy().sum()

        # c_matrix[1][1], # tp
        # c_matrix[1][0], # fn
        # c_matrix[0][1], # fp
        # c_matrix[0][0], # tn

        _cm = [_raw_data['tn'], _raw_data['fp']],[_raw_data['fn'], _raw_data['tp']]
        plot_confusion_matrix(np.array(_cm), ['bp', 'con'], normalize=True)
        _fname = os.path.join(report_dir, 'cm-over-{}'.format(testname))
        plt.title('{}'.format(testname))
        plt.tight_layout()
        plt.savefig(_fname + '.pdf', bbox_inches='tight')
        plt.close('all')

        grp1.boxplot(column=['acc', 'p_bp', 'p_con', 'r_bp', 'r_con', 'f1_bp', 'f1_con'], fontsize=16)
        _fname = os.path.join(report_dir, 'cm-over-box-{}'.format(testname))
        plt.title('{}'.format(testname))
        plt.tight_layout()
        plt.savefig(_fname + '.pdf', bbox_inches='tight')
        plt.close('all')

    grp = prf1_df.groupby('domain')
    _handlers = []
    for i in grp.groups.keys():
        _data = grp.get_group(i)
        _fname = os.path.join(report_dir, 'meas-{}.tex'.format(i))
        with open(_fname, "w") as fout:
            fout.write("\\begin{{table}}[h!]\n\\caption{{Measurements for {}}}\n\\begin{{small}}\\scalebox{{0.6}}{{".format(i))
            fout.write(_data[['docid', 'testname', 'tp', 'fn', 'fp', 'tn', 'acc', 'p_bp', 'p_con', 'r_bp',
                              'r_con', 'f1_bp', 'f1_con']].sort(['docid',
                                               'testname']).to_latex(index=False))
            fout.write("\n}\\end{small}\n\\end{table}\n")

        plt.figure()

        #_data.boxplot(column=['tp', 'fn', 'fp', 'tn'])
        _raw_data = _data[['tp', 'fn', 'fp', 'tn']].copy().sum()
        _cm = [_raw_data['tp'], _raw_data['fn']],[_raw_data['fp'], _raw_data['tn']]
        plot_confusion_matrix(np.array(_cm), ['bp', 'con'], normalize=True)
        _fname = os.path.join(report_dir, 'cm-prf1-{}'.format(i))
        plt.title('{}'.format(i))
        plt.tight_layout()
        plt.savefig(_fname + '.pdf', bbox_inches='tight')
        plt.close('all')

        _cm_groups = _data.groupby('testname')
        for _cm_grp in sorted(_cm_groups.groups):
            _raw_data = _cm_groups.get_group(_cm_grp)[['tp', 'fn', 'fp', 'tn']].copy().sum()

            _cm = [_raw_data['tn'], _raw_data['fp']],[_raw_data['fn'], _raw_data['tp']]
            plot_confusion_matrix(np.array(_cm), ['bp', 'con'], normalize=True)
            _fname = os.path.join(report_dir, 'cm-prf1-{}-{}'.format(i, _cm_grp))
            plt.title('{} - {}'.format(i, _cm_grp))
            plt.tight_layout()
            plt.savefig(_fname + '.pdf', bbox_inches='tight')
            plt.close('all')

        _data.boxplot(column=['acc', 'p_bp', 'p_con', 'r_bp', 'r_con', 'f1_bp', 'f1_con'], fontsize=16)
        _fname = os.path.join(report_dir, 'all-prf1-{}'.format(i))
        plt.title('{}'.format(i))
        plt.tight_layout()
        plt.savefig(_fname + '.pdf', bbox_inches='tight')
        plt.close('all')

    agg_gt_df = pd.DataFrame(columns=['domain', 'cross_unique_id',
                                      'gt_bp_cnt', 'gt_con_cnt', 'dbce_nodes'])

    # load groundtruth stats
    for (domain, _c_u_ids) in agg_ctx.groupby('domain')['cross_unique_id'].unique().items():
        for _c_u_id in _c_u_ids:
            _gt = gt2dict(config, _c_u_id)
            _cnt = Counter(gt2dict(config, _c_u_id).values())
            _total_dbce_nodes = agg_acc_df.loc[(agg_acc_df['cross_unique_id'] == _c_u_id)]['dbce_nodes'].mean()
            agg_gt_df = agg_gt_df.append(pd.DataFrame([[domain,
                                                        _c_u_id,
                                                        int(_cnt['dbce-class-bp']),
                                                        int(_cnt['dbce-class-content']),
                                                        int(_total_dbce_nodes)]],
                                                      columns=['domain', 'cross_unique_id',
                                                               'gt_bp_cnt', 'gt_con_cnt', 'dbce_nodes']))


    agg_gt_df['perc'] = (100*(agg_gt_df['gt_bp_cnt']+agg_gt_df['gt_con_cnt']))/agg_gt_df['dbce_nodes']

    plt.close('all')

    plt.figure()
    #plt.title('DBCE Node Statistics')
    agg_gt_df[['domain', 'gt_bp_cnt', 'gt_con_cnt', 'dbce_nodes']].boxplot(column=['dbce_nodes',
                                                                                   'gt_bp_cnt',
                                                                                   'gt_con_cnt'],
                                                                           by='domain',
                                                                           rot=45,
                                                                           fontsize=16,
                                                                           figsize=(10,5),
                                                                           layout=(1,3))
    # get rid of "Boxplot grouped by"
    plt.suptitle("")
    plt.tight_layout()
    _fname = os.path.join(report_dir, 'all-dbce-stats.pdf')
    plt.savefig(_fname)
    plt.close('all')

    plt.figure(figsize=plt.figaspect(1))
    _ann_nodes = agg_gt_df[['gt_bp_cnt', 'gt_con_cnt']].sum()
    _ann_nodes['missing'] = agg_gt_df['dbce_nodes'].sum() - _ann_nodes['gt_bp_cnt'] - _ann_nodes['gt_con_cnt']
    _ann_nodes.name = 'Annotations'
    _ann_nodes.plot.pie(autopct='%0.2f%%', fontsize=16, colormap='Paired', labels=['boilerplate', 'content', 'missing'])
    _fname = os.path.join(report_dir, 'all-dbce-ann-nodes.pdf')
    plt.savefig(_fname)
    plt.close('all')
    return []

def gt2dict(config, cross_unique_id):
    groundtruth = {}
    try:
        for (gt_hash, gt_xpath, gt_cls) in load_groundtruth_for_cross(
                config, cross_unique_id):
            groundtruth[gt_hash] = gt_cls
    except FileNotFoundError:
        return None

    return groundtruth

def plot_confusion_matrix(c_matrix, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    _c_matrix = c_matrix
    plt.imshow(c_matrix, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)

    if normalize:
        c_matrix = np.around(c_matrix.astype('float') / c_matrix.sum(axis=1)[:, np.newaxis], decimals=3)
    else:
        print('Confusion matrix, without normalization')

    if normalize:
        thresh = _c_matrix.max() * 0.9
    else:
        c_matrix.max() / 2

    for i, j in itertools.product(range(c_matrix.shape[0]), range(c_matrix.shape[1])):
        plt.text(j, i, "{} ({})".format(c_matrix[i, j], int(_c_matrix[i, j])),
                 horizontalalignment="center",
                 color="white" if _c_matrix[i, j] > thresh else "black",
                 fontsize=16)

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    # a 4307.0 871.0 78.0 606.0
    tp = _c_matrix[1][1] # tp
    fn = _c_matrix[1][0] # fn
    fp = _c_matrix[0][1] # fp
    tn = _c_matrix[0][0] # tn
    _acc = acc(tp, tn, fp, fn)
    _p_con = prec(tp, fp)
    _r_con = recall(tp, fn)
    _f1_con = f1score(_p_con, _r_con)

    _p_bp = prec(tn, fn)
    _r_bp = recall(tn, fp)
    _f1_bp = f1score(_p_bp, _r_bp)

    plt.figtext(1,
                0.6,
                "Summary\n acc \n p_con \n r_con \n f1_con \n p_bp \n r_bp \n f1_bp", wrap=True,
                horizontalalignment='left', fontsize=14)

    plt.figtext(1.1,
                0.6,
                ":{} \n :{} \n :{} \n :{} \n :{} \n :{} \n :{}".format(_acc,
                                                                       _p_con,
                                                                       _r_con,
                                                                       _f1_con,
                                                                       _p_bp,
                                                                       _r_bp,
                                                                       _f1_bp), wrap=True,
                horizontalalignment='left', fontsize=14)



def f1score(p, r):
    return round(2 * ((p*r)/(p+r)), 3)

def recall(tp, fn):
    return round(tp/float(tp+fn), 3)

def prec(tp, fp):
    return round(tp/float(tp+fp), 3)

def acc(tp, tn, fp, fn):
    return round((tp+tn)/float(tp+tn+fp+fn), 3)
