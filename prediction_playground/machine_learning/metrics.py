import inspect
import numpy

import sklearn.metrics
from sklearn.preprocessing import LabelBinarizer


class BaseCalculator(object):
    mapping = {}

    def __init__(self, clf, X_test, y_test, *args, **kwargs):
        self.clf = clf
        self.X_test = X_test
        self._binarizer = LabelBinarizer()
        self._binarizer.fit(y_test)
        self.y_test = self._binarizer.transform(y_test)


    def apply_mapping(self, dict):
        for calc_name, model_name in self.mapping.items():
            dict[model_name] = dict[calc_name]
            del dict[calc_name]
        return dict

    def base_calculations(self):
        raise NotImplementedError("You must override this method")

    def calculate(self):
        kwargs = self.base_calculations()
        result = {}
        for member in inspect.getmembers(self, inspect.isfunction):
            if self.is_calculate_function(member[0]):
                result[member[0].replace("calculate_", "")] = member[1](self, **kwargs)
        return self.apply_mapping(result)

    def is_calculate_function(self, member_name):
        return member_name.startswith("calculate_")


class SKLearnCalculator(BaseCalculator):

    _needed_args = set()
    _to_pred = ["y_pred", "y2"]
    _to_proba = ["y_score", "y_prob", 'probas_pred']
    _to_y_test = ["y_true", "y1"]

    def __init__(self, clf, X_test, y_test, MetricsClass=None):
        self.metrics_class = MetricsClass
        super(SKLearnCalculator, self).__init__(clf, X_test, y_test)
        self.add_calculate_functions()

    def _has_common_element(self, l1, l2):
        for x in l1:
            if x in l2:
                return True
        return False


    def base_calculations(self):
        result = {}
        if self._has_common_element(self._needed_args, self._to_pred):
            result["pred"] = self._binarizer.transform(self.clf.predict(self.X_test))
        if self._has_common_element(self._needed_args, self._to_proba):
            try:
                result["proba"] = self.clf.predict_proba(self.X_test)[:,-1]
            except:
                result["proba"] = self.clf.score(self.X_test, self.y_test)
        return result

    def add_calculate_functions(self):
        if self.metrics_class is None:
            predicate = lambda i: True
        else:
            field_names = self._get_field_names()
            predicate = lambda j: j in field_names

        for member in inspect.getmembers(sklearn.metrics,
                                         lambda x: inspect.isfunction(x) and not x.__name__.startswith("_") and predicate(x.__name__)):
            setattr(self, "calculate_" + member[0], self._add_wrapper(member[1]))
        return dict

    def _get_field_names(self):
        result = []
        for x in self.metrics_class._meta.get_fields():
            if hasattr(x, "related_model"):
                if hasattr(x.related_model, "alternative_model_name"):
                    result += [x.related_model.alternative_model_name]
                    continue
            result += [x.name]
        return result

    def _add_wrapper(self, func):
        args = inspect.getargspec(func)[0]
        self._needed_args.update(args)
        def wrapper(self, **kwargs):
            new_args = self._convert_args(args, **kwargs)
            return func(*new_args)
        return wrapper


    def _convert_args(self, args, **kwargs):
        new_args = []
        for arg in args:
            if arg in self._to_y_test:
                new_args += [self.y_test]
            elif arg in self._to_pred:
                new_args += [kwargs["pred"]]
            elif arg in self._to_proba:
                new_args += [kwargs["proba"]]
        return new_args
