import json
import os
import subprocess
import sys
from collections import Sequence

from django.conf import settings

from prediction_playground.models import PredictionPipeline, Prediction, BaseDataEntry


def start_prediction(pipeline_id, entries, strategy="batch"):
    if not isinstance(entries, Sequence):
        entries = [entries]
    if strategy == "direct":
        return predict(pipeline_id, entries)
    else:
        try:
            subprocess.call(["batch"])
            os.system(
                "echo '%s %s %s %s %s' | batch" %
                (
                    sys.executable, settings.BASE_DIR.join("manage.py"),
                    "predict",
                    "--pipeline=%s" % pipeline_id,
                    ' '.join(map(str, entries))
                )
            )
            return None
        except OSError:
            return predict(pipeline_id, entries)



def predict(pipeline_id, entries=None):
    pp = PredictionPipeline.objects.get(pk=pipeline_id)
    if not isinstance(entries, BaseDataEntry):
        new_entries = pp.dataframe.get_entries()
        if isinstance(entries, Sequence):
            entries = new_entries.filter(id__in=entries)
        else:
            entries = new_entries
    predictions = pp.sk_pipeline.to_object().predict(entries.to_df())
    for prediction, entry in zip(predictions, entries):
        Prediction.objects.create(entry=entry, sk_pipeline=pp.sk_pipeline, data=json.dumps(prediction))
        ppf  = pp.get_facade()
        if ppf is not None:
            for messager in ppf.messagers:
                messager().send_prediction(prediction, pp.get_facade().user_class.objects.get(id=entry.user.id))
    return predictions

