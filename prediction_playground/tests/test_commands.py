# -*- coding: utf-8 -*-
from __future__ import unicode_literals

import filecmp
import json
import os

from django.conf import settings
from django.core.management import call_command
# Create your tests here.
from sklearn.dummy import DummyClassifier

from prediction_playground.machine_learning import train
from prediction_playground.messaging.firebase import FirebaseMessager
from prediction_playground.models import DataframeModel, SKPipeline, PredictionPipeline
from prediction_playground.tests import PredictionPlaygroundTestCase


class CommandTestCase(PredictionPlaygroundTestCase):

    def test_import_csv(self):
        """Test if csv's can be imported"""
        call_command("import_dataset", "--dataframe=forever_alone_test", "--drop_columns=time", self.dataset_path)
        df = DataframeModel.objects.get(name="forever_alone_test").to_df()
        # Needed for some strange reason --> django autconverts to unicode?
        df = df.applymap(lambda x: x.encode('utf-8') if isinstance(x, unicode) else x)
        self.assertTrue("time" not in df.columns)
        self.dataset = self.dataset.drop("time", axis=1)
        self.assertEqual(self.dataset.shape, df.shape)
        self.assertEqual(self.dataset.columns.all(), df.columns.all())
        self.assertTrue(self.dataset.equals(df))


    def test_export_csv(self):
        """Test if csv's can be exported"""
        call_command("import_dataset", "--dataframe=forever_alone2", self.dataset_path)
        dir = os.path.join(settings.BASE_DIR, "prediction_playground/tests/extra_files")
        os.chdir(dir)
        call_command("export_dataset", "--dataframe=forever_alone2", "forever_alone2.csv")
        filepath = os.path.join(dir, "forever_alone2.csv")
        self.assertTrue(filecmp.cmp(self.dataset_path, filepath))
        call_command("export_dataset", "--dataframe=forever_alone2" , "--drop_columns=time", "forever_alone2.csv")
        self.assertFalse(filecmp.cmp(self.dataset_path, filepath))
        call_command("import_dataset", "--dataframe=forever_alone3", filepath)
        self.assertFalse("time" in DataframeModel.objects.get(name="forever_alone3").to_df().columns)

        os.remove(filepath)

    def test_predict(self):
        pp, sk_pipeline = self._create_pipeline(train=True)
        train(pp.id)
        call_command("predict", str(pp.id), str(pp.dataframe.get_entries()[0].id))

    def test_train(self):
        pp, sk_pipeline = self._create_pipeline()
        call_command("train", "--dataframe=kek", str(pp.id))



