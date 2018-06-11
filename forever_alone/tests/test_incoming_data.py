import json

from django.test import TestCase, Client, override_settings
from django.urls import reverse_lazy
from firebase_admin.messaging import ApiCallError

from forever_alone.models import FAMetrics
from prediction_playground.models import Prediction, DataframeModel


class TestWithData(TestCase):
    @override_settings(TESTING=True)
    def test_with_data(self):
        from forever_alone.machine_learning import FAFacade

        client = Client()
        pred_count = Prediction.objects.count()
        metrics_count = FAMetrics.objects.count()

        data = json.dumps(
            {"data": FAFacade().dataframe.get_entries().first().to_dict(), "firebase_id": "anon", "firebase_token": "123"}
        , default=str)
        with self.assertRaises(ApiCallError):
            request = client.post(reverse_lazy("fa_receive"), data, content_type="application/json")
            received_data = json.loads(request.content)
            print received_data
            self.assertTrue(received_data["success"])
            self.assertEqual(received_data["predictions"], [u'Yes'])
        self.assertEqual(Prediction.objects.count(), pred_count + 1)
        self.assertEqual(json.loads(Prediction.objects.last().data), u'Yes')
        self.assertEqual(Prediction.objects.last().entry.user.identifier, "anon")
        self.assertEqual(FAMetrics.objects.count(), metrics_count + 1)
        conf_matrix = FAMetrics.objects.last().confusion_matrix
        confusion_total = conf_matrix.tp + conf_matrix.tn + conf_matrix.fp + conf_matrix.fn
        self.assertGreater(confusion_total, 0)





#        self.assertGreater(FAFacade().sk_pipeline.prediction_set.count(), 0)