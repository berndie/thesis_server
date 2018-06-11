from django.core.management import BaseCommand

from prediction_playground.models import DataframeModel


class Command(BaseCommand):
    help = "Export the BaseDataEntry table as csv"

    def add_arguments(self, parser):
        parser.add_argument('--time_column', "-t", default=None, type=str, help="Store the creation time in this column")
        parser.add_argument('--drop_columns', '-c', nargs="+", default=None, type=str, help="The dummy_columns to be dropped")
        parser.add_argument('--dataframe', '-d', type=str, help="The name of the dataframe", required=True)
        parser.add_argument('path', type=str, help="Path to save the csv")

    def handle(self, *args, **options):
        dfm = DataframeModel.objects.get(name=options["dataframe"])
        if options["time_column"]:
            dfm.get_entries().values_list()
        else:
            df = dfm.to_df()


        if options["drop_columns"] is not None:
            df = df.drop(options["drop_columns"], axis=1)
        df.to_csv(options["path"], index=False)

