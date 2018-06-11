# -*- coding: utf-8 -*-
from __future__ import unicode_literals


# Create your models here.
from django.db import models
from django.db.models import PositiveIntegerField, CharField, DecimalField, OneToOneField, TextField
from django.utils import timezone

from prediction_playground.models import BaseDataEntry, BaseMLMetrics, MLMetricsQueryset, BaseDataEntryQueryset, \
    BetterDateTimeField, BaseMetricQueryset


class ForeverAloneEntry(BaseDataEntry):
    time = BetterDateTimeField(blank=True, null=True, default=timezone.now)
    gender = TextField(blank=True, null=True)
    sexuallity = TextField(blank=True, null=True)
    age = PositiveIntegerField(blank=True, null=True, default=0)
    income = TextField(blank=True, null=True)
    race = TextField(blank=True, null=True)
    bodyweight = TextField(blank=True, null=True)
    virgin = TextField(blank=True, null=True)
    prostitution_legal = TextField(blank=True, null=True)
    pay_for_sex = TextField(blank=True, null=True)
    friends = PositiveIntegerField(blank=True, null=True, default=0)
    social_fear = TextField(blank=True, null=True)
    depressed = TextField(blank=True, null=True)
    what_help_from_others = TextField(blank=True, null=True)
    attempt_suicide = TextField(blank=True, null=True)
    employment=  TextField(blank=True, null=True)
    job_title=  TextField(blank=True, null=True)
    edu_level=  TextField(blank=True, null=True)
    improve_yourself_how = TextField(blank=True, null=True)

    objects = BaseDataEntryQueryset.as_manager()


class ConfusionMatrixQueryset(BaseMetricQueryset):
    confusion_matrix_key= "confusion_matrix"
    def create_from_metric(self, values, model):
        return self.create(**dict(zip(("tn", "fp", "fn", "tp"), values.ravel())))

class ConfusionMatrix(models.Model):
    alternative_model_name = "confusion_matrix"
    tp = PositiveIntegerField()
    fp = PositiveIntegerField()
    tn = PositiveIntegerField()
    fn = PositiveIntegerField()
    objects = ConfusionMatrixQueryset.as_manager()




class FAMetrics(BaseMLMetrics):
    curves = ["roc_curve", "precision_recall_curve"]
    accuracy_score = DecimalField(max_digits=7, decimal_places=7)
    average_precision_score = DecimalField(max_digits=7, decimal_places=7)
    matthews_corrcoef = DecimalField(max_digits=7, decimal_places=7)
    cohen_kappa_score = DecimalField(max_digits=7, decimal_places=7)
    roc_auc_score = DecimalField(max_digits=7, decimal_places=7)
    f1_score = DecimalField(max_digits=7, decimal_places=7)
    confusion_matrix = OneToOneField("ConfusionMatrix")

    objects = MLMetricsQueryset.as_manager()