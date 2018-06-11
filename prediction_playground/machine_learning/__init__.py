import types
from collections import Sequence, Iterable

import pandas
from background_task import background
from background_task.models import Task
from django.db import transaction

from prediction_playground.helpers import import_facades
from prediction_playground.machine_learning.metrics import SKLearnCalculator
from prediction_playground.machine_learning.training import train, Trainer
from prediction_playground.models import PredictionPipeline, SKPipeline, DataframeModel, DataframeColumn, BaseDataEntry, \
    JSONDataEntry, BaseMLMetrics, PredictionSubject


class OverloadNeeded(object):
    FIELDS_TO_OVERLOAD = []

    @staticmethod
    def _check_method(new_class, field):
        method = new_class.__dict__.get("get_" + field, None)
        if method is None:
            return False
        else:
            return True

    def __new__(cls, *args, **kwargs):
        for field in cls.FIELDS_TO_OVERLOAD:
            if getattr(cls, field, None) is None and not cls._check_method(cls, field):
                raise TypeError("Can't instantiate '%s' because '(get_)%s' must be overridden" % (
                cls.__name__, field))
        return super(OverloadNeeded, cls).__new__(cls, *args, **kwargs)


class Singleton(object):
    __instance = None
    def __new__(cls, *args, **kwargs):
        if cls.__instance is None:
            cls.__instance = super(Singleton, cls).__new__(cls, *args, **kwargs)
        return cls.__instance


class PredictionPipelineFacade(Singleton, OverloadNeeded):
    FIELDS_TO_OVERLOAD = ["database_name", "dataframe_wrapper", "target_column_names", "sk_pipeline_object"]
    messagers = []
    trainer = Trainer
    trainer_kwargs = {}
    metrics_class = BaseMLMetrics
    calculator_class = SKLearnCalculator
    calc_metrics_mapping = {}
    chart_mapping = {}
    chart_layout = None
    train_interval = Task.DAILY
    user_class = PredictionSubject

    @property
    def _pp(self):
        return PredictionPipeline.objects.get(name=self.get_database_name())

    def __new__(cls, *args, **kwargs):

        new_obj = super(PredictionPipelineFacade, cls).__new__(cls, *args, **kwargs)
        new_obj.first_creation = True
        return new_obj

    def __init__(self):
        if self.first_creation:
            self.first_creation = False
            self._attach_pp()


    @property
    def id(self):
        return self._pp.id

    @property
    def dataframe(self):
        return self._pp.dataframe

    @property
    def sk_pipeline(self):
        return self._pp.sk_pipeline

    @property
    def target_columns(self):
        return self._pp.target_columns

    def _attach_pp(self):
        pp, created = PredictionPipeline.objects.get_or_create(name=self.get_database_name())
        tc = self.get_dataframe_wrapper().dfm.dataframecolumn_set.filter(name__in=self.get_target_column_names())
        if tc.count() != len(self.get_target_column_names()):
            err = [x for x in self.get_target_column_names() if x not in tc.values_list("name", flat=True)]
            raise TypeError("The target_columns %s are not related to dataframe %s" % (err, self.get_dataframe_wrapper().name))

        with transaction.atomic():
            pp.dataframe = self.get_dataframe_wrapper().dfm
            pp.save()
            pp.dataframe = self.get_dataframe_wrapper().dfm
            pp.target_columns.set(tc)
            if created:
                self.train.now(self)
            self.train(schedule=self.train_interval)


    def get_database_name(self):
        return self.database_name

    def get_messagers(self):
        return self.messagers

    def get_trainer_kwargs(self):
        return self.trainer_kwargs

    def get_sk_pipeline_object(self):
        return self.sk_pipeline_object

    def get_target_column_names(self):
        return self.target_column_names

    def get_dataframe_wrapper(self):
        return self.dataframe_wrapper

    @background
    def train(self):
        train(self.id, None,  TrainerClass=self.trainer, MLMetricsClass=self.metrics_class, CalculatorClass=self.calculator_class, calc_metric_mapping=self.calc_metrics_mapping)

class DataframeWrapper(object):

    def __init__(self, name="", DataEntryClass=BaseDataEntry, csv_path=None, time_column=None, datetime_format=None, columns=None):
        if not name:
            if isinstance(DataEntryClass, BaseDataEntry) and DataEntryClass is not BaseDataEntry and DataEntryClass is not JSONDataEntry:
                name = DataEntryClass.__name__
            else:
                raise TypeError("You need to give an explicit name, "
                                "or create your own subclass of %s "
                                "(The subclasses name will be used as name for the dataframe)" % BaseDataEntry)
        with transaction.atomic():
            self.dfm, created = self.get_dfm(name)
            if isinstance(csv_path, types.StringTypes):
                self.load_data(DataEntryClass, csv_path, created)
            elif isinstance(columns, Iterable):
                self.load_columns(columns)
            else:
                raise ValueError("You must specify either a csv_path or the columns for the dataframe")

    def load_data(self, DataEntryClass, csv_path, created):
        df = pandas.read_csv(csv_path)
        self.load_columns(df.columns)
        if DataEntryClass.objects.filter(dataframe=self.dfm).count() == 0:
            for index, row in df.iterrows():
                row = row.to_dict()
                DataEntryClass.objects.create(
                    dataframe=self.dfm,
                    **(row)
                )


    def load_columns(self, columns):
        for index,column in enumerate(columns):
            DataframeColumn.objects.get_or_create(name=column, dataframe=self.dfm, index=index)

    def get_dfm(self, name):
        return DataframeModel.objects.get_or_create(name=name)

    @property
    def id(self):
        return self.dfm.id

    @property
    def name(self):
        return self.dfm.name

    @property
    def columns(self):
        return self.dfm.dataframecolumn_set

    @property
    def entries(self):
        return  self.dfm.get_entries()



