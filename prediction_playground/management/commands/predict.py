from django.core.management import BaseCommand

from prediction_playground.machine_learning.predicting import predict


class Command(BaseCommand):
    help = "Train a new model"

    def add_arguments(self, parser):
        # parser.add_argument('--mlmodel', '-m', type=int, help="The id of the mlmodel")
        # parser.add_argument('--column_name', "-c", nargs="+",help="The target column name")
        # parser.add_argument('--predictor', '-p',type=str, help="The classname of the predictor")
        # parser.add_argument('--messager', '-e',type=str, help="The classname of the messager")
        # parser.add_argument('--preprocessor', '-o',type=str, help="The classname of the preprocessor")
        parser.add_argument('pipeline', type=int, help="The id of the pipeline")
        parser.add_argument('entries', nargs="+", default=None, type=long, help="Ids of the data entries")


    def handle(self, *args, **options):
        predict(options["pipeline"], options["entries"])

