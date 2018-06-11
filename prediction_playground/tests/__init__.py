import json
import os
from collections import Sequence

import pandas
from django.core.management import call_command

from django.test import TestCase
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.dummy import DummyClassifier
from sklearn.pipeline import Pipeline

from prediction_playground.messaging import BaseMessager, BaseMessage
from prediction_playground.messaging.firebase import FirebaseMessager
from prediction_playground.models import PredictionPipeline, SKPipeline, DataframeModel


class DummyTransformer(TransformerMixin):

    def fit(self, X, y=None, **fit_params):
        return self

    def transform(self, X, y=None, **fit_params):
        return [range(len(X)) for x in range(len(X))]

    def fit_transform(self, X, y=None, **fit_params):
        return super(DummyTransformer, self).fit_transform( X, y=y, **fit_params)



class PredictionPlaygroundTestCase(TestCase):
    extra_files_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "extra_files")
    dataset_path = os.path.join(extra_files_path, "forever_alone.csv")
    clean_dataset_path = os.path.join(extra_files_path, "forever_alone_clean.csv")
    dataset = pandas.read_csv(dataset_path)
    clean_dataset = pandas.read_csv(clean_dataset_path)
    clf = DummyClassifier(random_state=0)
    clf.fit(clean_dataset.drop("attempt_suicide", axis=1), clean_dataset["attempt_suicide"])
    dummy_sk_pipeline = Pipeline([("dummy_transformer", DummyTransformer()), ("dc", DummyClassifier(random_state=0))])

    def _create_pipeline(self, train=False):
        call_command("import_dataset", "--dataframe=kek", self.dataset_path)
        PredictionPipeline.objects.all().delete()
        SKPipeline.objects.all().delete()
        sk_pipeline = SKPipeline.objects.create()
        obj = self.dummy_sk_pipeline

        sk_pipeline.set_file(obj)
        sk_pipeline.save()
        pp = PredictionPipeline(
            name="top",
            sk_pipeline=sk_pipeline,
            dataframe=DataframeModel.objects.get(name="kek"),
        )
        pp.save()
        pp.target_columns.add(DataframeModel.objects.get(name="kek").dataframecolumn_set.get(name="attempt_suicide"))
        return pp, sk_pipeline

class DummyMessager(BaseMessager):
    def send_message(self, message):
        pass

    def send_prediction(self, prediction, user):
        self.send_message(BaseMessage("","","",prediction))

