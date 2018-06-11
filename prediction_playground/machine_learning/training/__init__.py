import pandas
from background_task import background
from sklearn.model_selection import train_test_split

from prediction_playground.machine_learning.metrics import SKLearnCalculator
from prediction_playground.models import PredictionPipeline, SKPipeline, DataframeModel, BaseDataEntry, BaseMLMetrics


class Trainer(object):

    def __init__(self, prediction_pipeline):
        self.pp = prediction_pipeline

    def train(self, entries=None):
        if isinstance(entries, BaseDataEntry):
            entries = entries
        elif entries is None:
            entries = self.pp.dataframe.get_entries()
        else:
            try:
                entries = self.pp.dataframe.get_entries().filter(id__in=entries)
            except (ValueError, TypeError):
                raise ValueError("'entries' must be a list of ids or a queryset of DataEntries! (%s found)" % str(type(entries)))


        df = entries.to_df()
        if df.empty:
            raise ValueError("The provided dataframe is empty")


        X = df.drop(self.pp.get_target_columns(), axis=1)
        y = df[self.pp.get_target_columns()]
        try:
            sk_pipeline = self.pp.sk_pipeline.to_object()
        except AttributeError:
            sk_pipeline = self.pp.get_facade().sk_pipeline_object
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, random_state=0)
        sk_pipeline.fit(self.X_train, self.y_train)
        return sk_pipeline, entries

def train(pipeline_id, entries=None, force=False, update_pipeline=True,
          TrainerClass=Trainer, MLMetricsClass=None, CalculatorClass=None, calc_metric_mapping=None):
    pp = PredictionPipeline.objects.get(pk=pipeline_id)
    trainer = TrainerClass(pp)
    clf, entries = trainer.train(entries)
    mlmetrics = None
    if MLMetricsClass is not None and CalculatorClass:
        calc = CalculatorClass(clf, trainer.X_test, trainer.y_test, MLMetricsClass)
        mlmetrics = MLMetricsClass.objects.save_calculations(calc_metric_mapping, **calc.calculate())


    if update_pipeline:
        new_sk_pipeline = SKPipeline(
            previous_version=pp.sk_pipeline
        )
        new_sk_pipeline.set_file(clf)
        new_sk_pipeline.save()
        if mlmetrics is not None:
            mlmetrics.sk_pipeline = new_sk_pipeline
            mlmetrics.save()
        new_sk_pipeline.entries.set(entries)

        pp.sk_pipeline = new_sk_pipeline
        pp.save()