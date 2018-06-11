import json

import pandas
from django.apps import apps
from django.db import models

# Managers
from django.db.models import ForeignObject
from django.db.models.fields.related import RelatedField, OneToOneField
from django.db.models.fields.reverse_related import ForeignObjectRel

from prediction_playground.models.inheritance import InheritanceQuerySet


class FetchOneQueryset(models.QuerySet):
    """Manager that extends the get functionality with a get_one"""

    def get_one(self, id=None, order_by=None, reverse=False, **kwargs):
        """Return an instance of a model"""
        qs = None
        if id is not None:
            return self.get(id)
        elif order_by is not None:
            qs = self.order_by(order_by)
        elif len(kwargs) > 0:
            qs = self.filter(**kwargs)
        else:
            qs = self.order_by("id").reverse()

        if reverse:
            qs = qs.reverse()
        return qs[0] if len(qs) > 0 else None


class DataConcatenateManager(models.QuerySet):

    def to_list(self, column_mapping=None):
        column = "data"
        lst = []
        extra_columns = []
        if isinstance(column_mapping, dict):
            extra_columns = column_mapping.keys()
        data = self.filter(**{column + "__isnull": False}).values(column, *extra_columns)
        for el in data:
            lst += [json.loads(el[column])]
            # Add extra data from BaseDataEntry
            if isinstance(column_mapping, dict):
                for column_name, output_name in column_mapping.items():
                    lst[-1][output_name] = el[column_name]
        return lst



class BaseDataEntryQueryset(FetchOneQueryset, InheritanceQuerySet):

    def to_df(self, column_mapping=None):
        column_names = self.first().dataframe.dataframecolumn_set
        if isinstance(column_mapping, dict):
            column_names.filter(name__not_in=[name for name in self.model.exclude_fields if name not in column_mapping.keys()])
        column_names = column_names.values_list("name", flat=True)
        if isinstance(column_mapping, dict):
            column_names_mapped = [column_mapping[name] if name in column_mapping else name for name in column_names]
        else:
            column_names_mapped = column_names

        df = pandas.DataFrame.from_records(list(self.values_list(*column_names)), columns=column_names_mapped)
        return df.applymap(lambda x: x.encode('utf-8') if isinstance(x, unicode) else x)

class JSONDataEntryQueryset(BaseDataEntryQueryset, DataConcatenateManager):
    def to_df(self, column_mapping=None):
        if len(self):
            df = pandas.DataFrame(self.to_list(column_mapping))
            return df.applymap(lambda x: x.encode('utf-8') if isinstance(x, unicode) else x)
        return pandas.DataFrame()



class CurveQueryset(FetchOneQueryset):
    def create_curve(self, data):
        curve_points = []
        for points in zip(*data):
            curve_points += [self.create(**dict(zip(self.model.field_order, points)))]
        return curve_points

class MLMetricsQueryset(FetchOneQueryset):
    def save_calculations(self, mapping=None, **kwargs):
        kwargs = self._apply_mapping(mapping, **kwargs)
        related_fields, kwargs = self._prepare_related(**kwargs)
        obj = self.create(**kwargs)
        self._set_related(obj, related_fields)
        return obj


    def _apply_mapping(self, mapping, **kwargs):
        if mapping is None: return kwargs
        for key, new_key in mapping.items():
            kwargs[new_key] = kwargs[key]
            del kwargs[key]
        return kwargs

    def _prepare_related(self, **kwargs):
        related_fields = {}
        for field in [x for x in self.model._meta.get_fields() if isinstance(x, RelatedField) or isinstance(x, ForeignObjectRel)]:
            if hasattr(field.related_model, "alternative_model_name"):
                field_name = field.related_model.alternative_model_name
            else:
                field_name = field.name
            if field_name in kwargs:
                if isinstance(field, OneToOneField):
                    kwargs[field_name] = field.related_model.objects.create_from_metric(kwargs[field_name], self.model)
                else:
                    related_fields[field_name] = [field, kwargs[field_name]]
                    del kwargs[field_name]
        return related_fields, kwargs


    def _set_related(self, obj, related_fields):
        for field_name, lst in related_fields.items():
            metrics = lst[0].related_model.objects.create_from_metric(lst[1], obj)
            try:
                setattr(obj, field_name, metrics)
            except:
                getattr(obj, field_name).set(metrics)

class BaseMetricQueryset(FetchOneQueryset):
    def create_from_metric(self, values, model):
        """model can be class or object"""
        raise NotImplementedError("You must override this method")

class ThresholdCurveQueryset(BaseMetricQueryset):
    field_order = []
    def create_from_metric(self, values, model):
        result = []
        for args in zip(*values):
            result += [self.create(metrics=model,**dict(zip(self.field_order, args)))]
        return result

class PRCurveQueryset(ThresholdCurveQueryset):
    field_order = ["precision", "recall", "threshold"]

class ROCCurveQueryset(ThresholdCurveQueryset):
    field_order = ["fpr", "tpr", "threshold"]
