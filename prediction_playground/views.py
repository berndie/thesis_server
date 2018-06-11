# -*- coding: utf-8 -*-

from __future__ import unicode_literals

import inspect
import json

# Create your views here.
import numpy

import pandas
import types
from collections import Sequence, Counter

import django.dispatch
import firebase_admin
from django.conf import settings
from django.db.models.fields.related import RelatedField
from django.forms import model_to_dict
from django.http import JsonResponse, HttpResponseNotFound
# Create your views here.
from django.urls import reverse_lazy
from django.views import View
from django.views.generic import ListView, FormView, TemplateView
from firebase_admin import credentials, auth

from prediction_playground.helpers import get_all_messagers, get_subclasses, import_facades
from prediction_playground.machine_learning import PredictionPipelineFacade
from prediction_playground.machine_learning.predicting import start_prediction
from prediction_playground.models import BaseDataEntry, FirebaseUser, DataframeModel, SKPipeline, JSONDataEntry, \
    PredictionSubject, Prediction
from prediction_playground.models.models import PredictionPipeline


def init_firebase():
    cred = credentials.Certificate(settings.PREDICTION_PLAYGROUND_FIREBASE_CERT_LOCATION)
    firebase_admin.initialize_app(cred)


class BaseReceiveView(View):
    user_class = PredictionSubject
    data_entry_class = BaseDataEntry
    prediction_strategy = "batch"
    dataframe = None

    def fail(self, e):
        return JsonResponse({"success": False, "error": str(e)})

    def parse_dataframe(self, dataframe=None, **kwargs):
        if dataframe is not None:
            self.dataframe = dataframe
        elif isinstance(self.dataframe, types.StringTypes):
            self.dataframe = DataframeModel.objects.get(name=self.dataframe)
        return self.dataframe

    def _parse_data(self, request, *args, **kwargs):
        parsed_data = self.parse_data(request, *args, **kwargs)
        self.parse_dataframe(**parsed_data)
        user = self.get_user(**parsed_data)
        return self.save_data(user, **parsed_data)

    def parse_data(self, request, *args, **kwargs):
        message = json.loads(request.body)
        result = {
            "data": message["data"],
            "identifier": message.get("user", None)
        }
        try:
            result["dataframe"] = DataframeModel.objects.get(name=message["dataframe"])
        except DataframeModel.DoesNotExist:
            pass
        return result

    def save_data(self, user, data=None, **kwargs):

        if isinstance(data, Sequence):
            entries = []
            for entry_data in data:
                entries += [self.data_entry_class.objects.create(dataframe=self.dataframe, user=user, **entry_data).id]
        else:
            return [self.data_entry_class.objects.create(dataframe=self.dataframe, user=user, **data).id]

    def _start_predictions(self, data_entries):
        for pp in PredictionPipeline.objects.filter(dataframe=self.dataframe):
            return start_prediction(pp.id, data_entries, self.prediction_strategy)

    def get_user(self, identifier=None, **kwargs):
        if identifier is not None:
            return self.user_class.objects.get_or_create(identifier=identifier)[0]
        else:
            return None

    def post(self, request, *args, **kwargs):
        if self.data_entry_class not in get_subclasses(BaseDataEntry): raise TypeError(
            "You must specify a valid data_entry_class")
        # try:
        de = self._parse_data(request, *args, **kwargs)
        predictions = self._start_predictions(de)
        if predictions is None:
            return JsonResponse({"success": True})
        else:
            return JsonResponse({"success": True, "predictions": list(predictions)})
        # except Exception as e:
        #     return self.fail(e)


class JSONReceiveView(BaseReceiveView):
    data_entry_class = JSONDataEntry


# class GDriveReceiveView(BaseReceiveView):
#     def post(self, request, *args, **kwargs):
#         data = json.loads(request.POST["data"])
#         BaseDataEntry.objects.create_entry_from_dict(data, "time")
#         super(GDriveReceiveView, self).post(request, *args, **kwargs)


class FirebaseReceiveView(BaseReceiveView):

    def verification(self, token):
        # Throws error if user is not valid
        firebase_admin.auth.verify_id_token(token)

    def parse_data(self, request, *args, **kwargs):
        message = json.loads(request.body)
        result = {
            "data": message["data"],
            "identifier": message["firebase_id"],
            "firebase_token": message["firebase_token"]
        }

        # Throws error if user is not valid
        self.verification(result["identifier"])
        return result

    def get_user(self, identifier=None, **kwargs):
        return FirebaseUser.objects.get_or_create(identifier=identifier, firebase_token=kwargs["firebase_token"])[0]

class GChart(object):
    def __init__(self, name, values, x_label="Time"):
        self.x_label = x_label
        self.name = name
        self.verbose_name = name.replace("_"," ").title()
        self.values = values
        self.package = ""
        self.gclass = None
        self.extra_options = ""

    def create_extra_options(self, dct):
        return json.dumps(dct)[1:-1]

class GLineChart(GChart):
    def __init__(self, name, values, x_label="Time"):
        super(GLineChart, self).__init__(name, values, x_label)
        self.package = "line"
        self.gclass = "LineChart"

class GDonutChart(GChart):
    def __init__(self, name, values, x_label="Prediction classes"):
        super(GDonutChart, self).__init__(name, values, x_label)
        self.gclass = "PieChart"
        self.extra_options = self.create_extra_options({"piehole": 0.4})



class GChartFactory(object):
    mapping = {"line": GLineChart}

    @classmethod
    def create(cls, name, values, chart_type=None, **kwargs):
        if chart_type is None: chart_type = "line"
        if inspect.isclass(chart_type):
            ChartClass = chart_type
        else:
            ChartClass = cls.mapping[chart_type]
        return ChartClass(name, values, **kwargs)




class OverviewChartLayout(object):
    def __init__(self, matrix=None):
        self.matrix = matrix
        self.index = 0
    def __iter__(self):
        self.index = 0
        return self

    def next(self):
        if self.index < len(self.matrix):
            self.index += 1
            return (self.matrix[self.index -1], 12/len(self.matrix[self.index -1]))
        else:
            raise StopIteration



#overview_import_check = False
class OverviewView(TemplateView):
    template_name = "prediction_playground/overview.html"
    overview_import_check = False
    # Prediction pipeline

    pp = None

    def get(self, request, pipeline_name=None, **kwargs):
        self.determine_prediction_pipeline(pipeline_name)
        if self.pp is None:
            return HttpResponseNotFound("This pipeline with name '%s' does not exist" % pipeline_name)
        else:
            return super(OverviewView, self).get(request,**kwargs)

    def determine_prediction_pipeline(self, pipeline_name):
        import_facades()
        try:
            self.pp = PredictionPipeline.objects.get(name=pipeline_name)
        except PredictionPipeline.DoesNotExist:
            pass




    def create_charts(self, context):
        ppf = self.pp.get_facade()
        context["charts"] = []
        context["chart_packages"] = set()
        context["special_charts"] = []
        if ppf is not None:
            metrics, entries_count =self._get_all_metrics()
            df = pandas.DataFrame(metrics)
            # df = df.astype("float", errors="ignore")
            for column in [x for x in df.columns if x != "created_on"]:
                context["charts"] += [GChartFactory.create(column, df.as_matrix(["created_on", column]), chart_type=ppf.chart_mapping.get("column", None))]
                context["chart_packages"].update([str(context["charts"][-1].package)])
            context = self._prediction_count_chart(context)
            context["special_charts"] += [GLineChart("data_count", zip(df.loc[:, "created_on"], entries_count))]

        context["chart_packages"] = [str("corechart")] + list(context["chart_packages"])
        context["all_charts"] = context["special_charts"] + context["charts"]
        return context

    def _prediction_count_chart(self, context):
        pred = Prediction.objects.filter(entry__in=self.pp.dataframe.get_entries()).values_list("data", flat=True)
        c = Counter()
        for x in pred:
            if not isinstance(x, types.StringTypes) and isinstance(x, Sequence):
                c.update(x)
            else:
                c[x] += 1

        context["special_charts"] += [GDonutChart("prediction_distribution", c.items())]
        return context







    def create_layout(self, context):
        context["chart_layout"] = self.pp.get_facade().chart_layout
        if context["chart_layout"] is None:
            charts = [x.name for x in context["charts"]]
            if len(context["charts"]) % 2 > 0:
                prefix = [[charts[0]]]
                charts = charts[1:]
            else:
                prefix = []
                charts = charts
            context["chart_layout"] = prefix + list(numpy.array(charts).reshape(len(charts)/2, 2))
        context["chart_layout"] = OverviewChartLayout(context["chart_layout"])
        return context





    def _get_all_metrics(self):
        skp = self.pp.sk_pipeline
        metrics_class = self.pp.get_facade().metrics_class
        metrics_list = []
        entries_count = []
        iter_count = 0
        while skp is not None and iter_count < self.kwargs.get("limit", 10):
            entries_count += [skp.entries.count()]
            metrics = metrics_class.objects.get(id=skp.metrics.id)
            rel_field_names = [x.name for x in metrics._meta.get_fields() if isinstance(x, RelatedField)]
            dct = model_to_dict(metrics, exclude=["id","sk_pipeline", "basemlmetrics_ptr"] + rel_field_names)
            dct["created_on"] = metrics.created_on
            metrics_list += [dct]
            iter_count += 1
            skp = skp.previous_version
        return metrics_list, entries_count

    def create_pipeline_list(self, context):
        context["pipeline_list"] = PredictionPipeline.objects.all()
        context["pipeline_list_active"] = self.pp
        return context

    def get_context_data(self, **kwargs):
        context = super(OverviewView, self).get_context_data(**kwargs)
        context = self.create_charts(context)
        context = self.create_layout(context)
        context = self.create_pipeline_list(context)
        return context


