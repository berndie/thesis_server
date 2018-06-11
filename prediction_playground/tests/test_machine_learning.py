import numpy

import pandas
from sklearn.dummy import DummyClassifier
from sklearn.exceptions import NotFittedError
from sklearn.pipeline import Pipeline

from prediction_playground.machine_learning import PredictionPipelineFacade, DataframeWrapper
from prediction_playground.machine_learning.predicting import predict
from prediction_playground.machine_learning.training import Trainer
from prediction_playground.messaging import BaseMessager
from prediction_playground.models import DataframeModel, JSONDataEntry
from prediction_playground.tests import PredictionPlaygroundTestCase, DummyTransformer


class PredictionTestCase(PredictionPlaygroundTestCase):
    def setUp(self):
        self.pp, self.skp = self._create_pipeline()

    def test_prediction(self):
        df = DataframeModel.objects.get(name="kek").to_df()

        self.dummy_sk_pipeline.fit(df.drop("attempt_suicide", axis=1), df["attempt_suicide"])
        p1 = self.dummy_sk_pipeline.predict(df.drop("attempt_suicide", axis=1))
        with self.assertRaises(NotFittedError):
            p2 = predict(self.pp.id)
        new = self.pp.sk_pipeline.to_object()
        new.fit(df.drop("attempt_suicide", axis=1), df["attempt_suicide"])
        self.pp.sk_pipeline.set_file(new)
        self.pp.sk_pipeline.save()
        p2 = predict(self.pp.id)

        self.assertTrue(numpy.array_equal(p1,p2))


    def test_messager(self):
        with self.assertRaises(NotFittedError):
            predict(self.pp.id)
        df = DataframeModel.objects.get(name="kek").to_df()
        new = self.pp.sk_pipeline.to_object()
        new.fit(df.drop("attempt_suicide", axis=1), df["attempt_suicide"])
        self.pp.sk_pipeline.set_file(new)
        self.pp.sk_pipeline.save()
        p2 = predict(self.pp.id)


class MetricsCalculatorTestCase(PredictionPlaygroundTestCase):
    pass


class TrainerTestCase(PredictionPlaygroundTestCase):
    def setUp(self):
        self.pp, self.skp = self._create_pipeline()

    def test_empty_train(self):
        empty_trainer = Trainer(self.pp)
        with self.assertRaisesRegexp(ValueError, "empty"):
            clf, entries = empty_trainer.train([])

    def test_faulty_train(self):
        empty_trainer = Trainer(self.pp)
        with self.assertRaisesRegexp(ValueError, "entries"):
            clf, entries = empty_trainer.train("TOPKEK")


    def test_train(self):
        trainer = Trainer(self.pp)
        clf, entries = trainer.train()
        pred = clf.predict(self.pp.dataframe.to_df())
        self.assertEqual(len(pred), pandas.read_csv(self.dataset_path).shape[0])


class FacadeTest(PredictionPlaygroundTestCase):

    def test_missing_properties(self):
        with self.assertRaises(TypeError):
            class Facaderino(PredictionPipelineFacade):
                pass
            Facaderino()

    def test_methods(self):
        class DummyMethodFacade(PredictionPipelineFacade):
            def get_database_name(self): return "hallo"
            def get_views(self): return []
            def get_dataframe_wrapper(self): return None
            def get_target_column_names(self): return []
            def get_sk_pipeline_object(self): return "hallo"

    def test_working_dummy(self):
        class DummyFacade(PredictionPipelineFacade):
            database_name = "dummy"
            views = []
            dataframe_wrapper = DataframeWrapper("hallo", JSONDataEntry, PredictionPlaygroundTestCase.dataset_path)
            target_column_names = ["attempt_suicide"]
            sk_pipeline_object = Pipeline([("dummy_transformer", DummyTransformer()), ("dc", DummyClassifier(random_state=0))])


class DataframeWrapperTest(PredictionPlaygroundTestCase):
    def test_only_name(self):
        with self.assertRaises(ValueError):
            dw = DataframeWrapper("a")

    def test_empty_constructor(self):
        with self.assertRaises(TypeError):
            dw = DataframeWrapper()

    def test_with_columns(self):
        dw = DataframeWrapper("b", columns=["a", "b"])
        self.assertEqual(list(dw.dfm.dataframecolumn_set.all().values_list("name", flat=True)), ["a", "b"])

    def test_with_data(self):
        dw = DataframeWrapper("c", JSONDataEntry, self.clean_dataset_path)
        self.assertEqual(dw.dfm.to_df().shape, pandas.read_csv(self.clean_dataset_path).shape)



