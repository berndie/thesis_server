# -*- coding: utf-8 -*-
from __future__ import unicode_literals


from django.apps import AppConfig




class PredictionPlaygroundConfig(AppConfig):
    name = 'prediction_playground'

    def ready(self):
        import signals

