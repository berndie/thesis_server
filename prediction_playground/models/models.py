# -*- coding: utf-8 -*-
from __future__ import unicode_literals

import json
import os
import pickle

import pandas
import sklearn
from django.conf import settings
from django.core.files.base import ContentFile
from django.db import models
from django.db.models import TextField, DateTimeField, ForeignKey, ManyToManyField, FileField, OneToOneField, \
    DecimalField, IntegerField, PositiveIntegerField
from django.db.models.fields.related import RelatedField
from django.forms import model_to_dict
from django.utils.timezone import now

# Models
from prediction_playground.helpers import find_subclass_by_name, ClassNotFoundException, get_subclasses
from prediction_playground.models import DatetimeParser
from prediction_playground.models.managers import BaseDataEntryQueryset, FetchOneQueryset, \
    JSONDataEntryQueryset, CurveQueryset, MLMetricsQueryset, PRCurveQueryset, ROCCurveQueryset
from prediction_playground.models.users import PredictionSubject


class CreatedOn(models.Model):
    created_on = DateTimeField(default=now, editable=False)

    def set_time(self, string, datetime_format=None):
        self.created_on = DatetimeParser.parse_time(string, datetime_format)

    class Meta:
        abstract = True


class FileModel(CreatedOn):
    sub_dir = ""
    file = FileField(upload_to=os.path.join(settings.PREDICTION_PLAYGROUND_DATA_DIRECTORY_NAME, sub_dir))

    def set_file(self, obj):
        name = "%s_%s" % (obj.__class__.__name__, self.created_on.strftime("%Y%m%d_%H%M%S"))
        self.file.save(name, ContentFile(pickle.dumps(obj)), save=False)

    def to_object(self):
        with open(self.file.path, "rb") as fp:
            return pickle.load(fp)

    def __unicode__(self):
        lst = os.path.basename(self.file.name).split("_")
        if len(lst) == 3:
            return "%s (%s)" % (lst[0], DatetimeParser.parse_time(lst[1] + lst[2], "%Y%m%d%H%M%S"))
        else:
            return "%s: No file yet!" % self.__class__.__name__

    class Meta:
        abstract = True


class DataframeModel(CreatedOn):
    name = TextField(unique=True)


    def _get_downcasted_entries_queryset(self):
        # Hack for downcasting the queryset
        return self.basedataentry_set.all()[:1].select_subclasses()[0].__class__.objects.filter(dataframe=self)

    def get_entries(self):
        return self._get_downcasted_entries_queryset()

    def to_df(self, column_mapping=None):
        entries = self._get_downcasted_entries_queryset()

        df = pandas.DataFrame(entries.to_df(column_mapping),
                              columns=self.dataframecolumn_set.all().order_by("index").values_list("name", flat=True))
        return df.applymap(lambda x: x.encode('utf-8') if isinstance(x, unicode) else x)


class DataframeColumn(CreatedOn):
    dataframe = ForeignKey("DataframeModel", on_delete=models.CASCADE)
    name = TextField()
    index = PositiveIntegerField()

    def to_df(self):
        return self.dataframe.to_df()[self.name]

    class Meta:
        unique_together = ('dataframe', 'name',)


class BaseDataEntry(CreatedOn):
    dataframe = ForeignKey("DataframeModel", on_delete=models.CASCADE)
    user = ForeignKey(PredictionSubject, null=True, blank=True)
    objects = BaseDataEntryQueryset.as_manager()

    def __init__(self, *args, **kwargs):
        if "created_on" in kwargs:
            self.set_time(kwargs["created_on"], kwargs.get("datetime_format", None))
            del kwargs["created_on"]
        if "datetime_format" in kwargs:
            del kwargs["datetime_format"]

        super(BaseDataEntry, self).__init__(*args, **kwargs)

    def to_df(self, column_mapping=None):
        return pandas.DataFrame(self.to_dict(column_mapping))

    def to_dict(self, column_mapping=None):
        fields = list(self.dataframe.dataframecolumn_set.values_list("name", flat=True))
        if isinstance(column_mapping, dict) and "created_on" in column_mapping:
            fields += ["created_on"]
        d = model_to_dict(self, fields=fields)
        if isinstance(column_mapping, dict):
            for old_column, new_column in column_mapping.items():
                d[new_column] = d[old_column]
                del d[old_column]
        return d

class JSONDataEntry(BaseDataEntry):
    data = TextField()
    objects = JSONDataEntryQueryset.as_manager()

    def __init__(self, *args, **kwargs):
        if "dataframe" in kwargs:
            column_names = kwargs["dataframe"].dataframecolumn_set.values_list("name", flat=True)
            data = {}
            for name in column_names:
                data[name] = kwargs.get(name, None)
                del kwargs[name]
            kwargs["data"] = json.dumps(data)
        super(JSONDataEntry, self).__init__(*args, **kwargs)

    def to_df(self, column_mapping=None):
        df = pandas.Series(self.to_dict(column_mapping)).to_frame().T
        return df.applymap(lambda x: x.encode('utf-8') if isinstance(x, unicode) else x)

    def to_dict(self, column_mapping=None):
        data = json.loads(self.data)
        for name, output_name in column_mapping.items():
            data[output_name] = data[name]
            del data[name]
        return data


class Prediction(CreatedOn):
    entry = ForeignKey("BaseDataEntry", on_delete=models.CASCADE)
    sk_pipeline = ForeignKey("SKPipeline", on_delete=models.CASCADE)
    data = TextField()


class SKPipeline(FileModel):
    sub_dir = "/mlmodels"
    entries = ManyToManyField("BaseDataEntry", blank=True)
    previous_version = ForeignKey("SKPipeline", on_delete=models.SET_NULL, blank=True, null=True)
    objects = FetchOneQueryset.as_manager()
    sklearn_version = TextField(default=sklearn.__version__, editable=False)


class BaseMLMetrics(CreatedOn):
    curves = tuple()
    objects = MLMetricsQueryset.as_manager()
    sk_pipeline = OneToOneField("SKPipeline", related_name="metrics", on_delete=models.CASCADE, blank=True, null=True)


class CurvePoint(CreatedOn):
    field_order = tuple()
    metrics = ForeignKey("BaseMLMetrics", on_delete=models.CASCADE)

    objects = CurveQueryset.as_manager()
    class Meta:
        abstract = True


class ThresholdCurvePoint(CurvePoint):
    threshold = DecimalField(decimal_places=1, max_digits=8)

    class Meta:
        abstract = True


class PrecisionRecallCurvePoint(ThresholdCurvePoint):
    alternative_model_name = "precision_recall_curve"
    precision = DecimalField(decimal_places=1, max_digits=8)
    recall = DecimalField(decimal_places=1, max_digits=8)
    objects = PRCurveQueryset.as_manager()


class ROCCurvePoint(ThresholdCurvePoint):
    alternative_model_name = "roc_curve"
    fpr = DecimalField(decimal_places=1, max_digits=8)
    tpr = DecimalField(decimal_places=1, max_digits=8)
    objects = ROCCurveQueryset.as_manager()



class PredictionPipeline(CreatedOn):
    name = TextField(unique=True)
    sk_pipeline = ForeignKey("SKPipeline", on_delete=models.SET_NULL, null=True, blank=True)
    dataframe = ForeignKey("DataframeModel",  on_delete=models.SET_NULL, null=True, blank=True)
    target_columns = ManyToManyField("DataframeColumn")

    def get_target_columns(self):
        lst = self.target_columns.all().values_list("name", flat=True)
        if len(lst) == 1:
            return lst[0]
        else:
            return lst

    def get_facade(self):
        from prediction_playground.machine_learning import PredictionPipelineFacade
        for cls in get_subclasses(PredictionPipelineFacade):
            if cls().database_name == self.name:
                return cls()
            else:
                return None
