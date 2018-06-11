import glob
import importlib
import inspect
import os

from django.apps import apps
from django.conf import settings
from sklearn.dummy import DummyClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

from prediction_playground.messaging import BaseMessager


def get_subclasses(cls):
    """returns all subclasses of argument, cls"""
    if issubclass(cls, type):
        subclasses = cls.__subclasses__(cls)
    else:
        subclasses = cls.__subclasses__()
    for subclass in subclasses:
        subclasses.extend(get_subclasses(subclass))
    return subclasses

def find_subclass_by_name(name, cls):
    for subcls in get_subclasses(cls):
        if unicode(name.lower().replace('"','')) == unicode(subcls.__name__.lower()):
            return subcls
    raise ClassNotFoundException("The class with name '%s' doesn't exist, did you rename it?" % name)



def _meta_getter(classes, as_text=True, as_tuples=True):
    lst = []
    for index, cls in enumerate(classes):
        tmp = cls
        if as_text and not isinstance(cls, str) and not isinstance(cls, unicode):
            tmp = cls.__name__
        if as_tuples:
            tmp = (index, tmp)
        lst += [tmp]
    return lst


def get_all_classifiers(as_text=True, as_tuples=True):
    return _meta_getter([
        DummyClassifier, KNeighborsClassifier, SVC, LinearSVC, GaussianProcessClassifier, DecisionTreeClassifier,
        RandomForestClassifier, AdaBoostClassifier, GaussianNB, QuadraticDiscriminantAnalysis, MLPClassifier
    ], as_text, as_tuples)


def get_all_dataframes(as_text=True, as_tuples=True):
    from prediction_playground.models import DataframeModel
    return _meta_getter(DataframeModel.objects.all().values_list("name", flat=True), as_text=as_text, as_tuples=as_tuples)

def get_all_dataframe_columns(dataframe, as_text=True, as_tuples=True):
    pass

def get_all_receive_views(base_view, as_text=True, as_tuples=True):
    return _meta_getter(get_subclasses(base_view), as_text=as_text, as_tuples=as_tuples)


def get_all_messagers(as_text=True, as_tuples=True):
    return _meta_getter(get_subclasses(BaseMessager), as_text=as_text, as_tuples=as_tuples)


class ClassNotFoundException(TypeError):
    pass




def import_facades():
    from prediction_playground.machine_learning import PredictionPipelineFacade
    for package in settings.INSTALLED_APPS:
        if package == "prediction_playground": continue
        try:
            module = importlib.import_module(package + ".machine_learning")
            for facade in inspect.getmembers(module, lambda x: x in get_subclasses(PredictionPipelineFacade)):
                facade[1]()
        except ImportError:
            pass