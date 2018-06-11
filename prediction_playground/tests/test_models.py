from prediction_playground.models import SKPipeline
from prediction_playground.tests import PredictionPlaygroundTestCase


class SKPipelineTest(PredictionPlaygroundTestCase):
    def test_delete(self):
        parent = SKPipeline.objects.create()
        child = SKPipeline.objects.create(previous_version=parent)
        second_child = SKPipeline.objects.create(previous_version=parent)
        child_of_child = SKPipeline.objects.create(previous_version=child)
        child_of_child2 = SKPipeline.objects.create(previous_version=child)


        child.delete()
        #second_child, child_of_child, child_of_child2
        self.assertEqual(parent.skpipeline_set.count(), 3)
        self.assertEqual(parent.skpipeline_set.get(id=child_of_child.id).id, child_of_child.id)
        self.assertEqual(SKPipeline.objects.get(id=child_of_child.id).previous_version.id, parent.id)
        parent.delete()
        self.assertIsNone(SKPipeline.objects.get(id=child_of_child.id).previous_version)

    def test_metrics(self):
        x = SKPipeline.objects.create()