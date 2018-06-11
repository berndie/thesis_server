from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler, LabelBinarizer

from forever_alone.machine_learning.preprocessors import FACleaner, FAImputer, FATransformer
from forever_alone.models import ForeverAloneEntry, FAMetrics
from prediction_playground.machine_learning import PredictionPipelineFacade, DataframeWrapper, SKLearnCalculator
from prediction_playground.messaging.firebase import FirebaseMessager
from prediction_playground.models import FirebaseUser


class FAFacade(PredictionPipelineFacade):
    database_name = "ForeverAlone"
    dataframe_wrapper = DataframeWrapper("forever_alone", ForeverAloneEntry, csv_path="forever_alone/tests/extra_files/forever_alone.csv")
    metrics_class = FAMetrics
    calculator_class = SKLearnCalculator
    target_column_names = ["attempt_suicide"]
    sk_pipeline_object = Pipeline([
        ("clean",FACleaner()),
        ("impute",FAImputer()),
        ("preprocess", FATransformer()),
        ("scale", RobustScaler()),
        ("clf",RandomForestClassifier(random_state=0)),

    ])
    messagers = [FirebaseMessager]
    user_class = FirebaseUser
