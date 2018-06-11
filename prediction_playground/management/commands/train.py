from django.core.management import BaseCommand

from prediction_playground.machine_learning.training import train
from prediction_playground.models import DataframeModel


class Command(BaseCommand):
    help = "Train a new model"

    def add_arguments(self, parser):
        parser.add_argument('-f', "--force",action='store_true' , help="Retrain even if there is no new data")
        parser.add_argument('-n', "--no-update", action='store_true', help="Don't update the pipeline containing the mlmodel")
        parser.add_argument('-e', '--entries', nargs="+", default=None, type=long, help="Ids of the data entries")
        parser.add_argument('-d', '--dataframe', type=str, help="Ids of the data entries")
        parser.add_argument('pipeline', type=int, help="The id of the pipeline to train")


        # parser.add_argument('dataframe', type=str, help="The name of the dataframe to train on")


    def handle(self, *args, **options):
        if options["dataframe"] is None and options["entries"] is None:
            train(options["pipeline"], None, not options["no_update"])
        elif options["entries"] is not None:
            train(options["pipeline"], options["entries"], not options["no_update"])
        elif options["dataframe"] is not None:
            entries = DataframeModel.objects.get(name=options["dataframe"]).get_entries()
            train(options["pipeline"], entries, not options["no_update"])
