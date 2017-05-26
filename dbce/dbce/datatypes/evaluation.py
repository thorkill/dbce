

class EvalResultContainer(dict):

    def add_result(self, domain, result):
        _ = self.setdefault(domain, [])
        _.append(result)

class EvalResult(object):

    def __init__(self, wlid):
        self.docid = wlid
        self.domain = None
        self.groundtruth = None
        self.result = None
