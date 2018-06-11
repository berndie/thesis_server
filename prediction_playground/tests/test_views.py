import json

from django.core.management import call_command
from django.test import Client, TestCase
from django.urls import reverse_lazy
from sklearn.dummy import DummyClassifier
from sklearn.pipeline import Pipeline

from prediction_playground.machine_learning import PredictionPipelineFacade, DataframeWrapper
from prediction_playground.models import DataframeModel, BaseDataEntry, JSONDataEntry
from prediction_playground.tests import PredictionPlaygroundTestCase, DummyTransformer
from prediction_playground.views import JSONReceiveView


class ReceiveViewTestCase(PredictionPlaygroundTestCase):
    def setUp(self):
        class TestFacade(PredictionPipelineFacade):
            database_name = "dummy"
            views = [JSONReceiveView]
            dataframe_wrapper = DataframeWrapper("hallo", JSONDataEntry, PredictionPlaygroundTestCase.dataset_path)
            target_column_names = ["attempt_suicide"]
            sk_pipeline_object = Pipeline([("dummy_transformer", DummyTransformer()), ("dc", DummyClassifier(random_state=0))])

    def test_json_receive_view(self):
        client =  Client()
        data = json.dumps({"data": json.loads(JSONDataEntry.objects.first().data), "user": "bernd", "dataframe":"hallo"})
        request = client.post(reverse_lazy("json_receive"), data, content_type="application/json")
        received_data = json.loads(request.content)
        print received_data
        self.assertTrue(received_data["success"])


class OverviewViewTest(TestCase):
    def test_overview(self):
        pass