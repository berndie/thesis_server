# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.apps import apps

from forever_alone.models import ForeverAloneEntry
from prediction_playground.models import DataframeModel
from prediction_playground.views import OverviewView, BaseReceiveView, FirebaseReceiveView


class WebView(OverviewView):
    metrics = ["accuracy"]


class FAReceiveView(FirebaseReceiveView):
    dataframe = "forever_alone"
    data_entry_class = ForeverAloneEntry

    def verification(self, token):
        if token != "anon":
            super(FAReceiveView, self).verification(token)
