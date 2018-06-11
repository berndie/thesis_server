import json

import pandas
from django.core.management import BaseCommand
from django.utils.timezone import now

from prediction_playground.helpers import find_subclass_by_name
from prediction_playground.models import BaseDataEntry, DataframeModel, DataframeColumn


class Command(BaseCommand):
    help = "Import an csv into the BaseDataEntry table"

    def add_arguments(self, parser):
        # parser.add_argument('--delimiter', '-d', default=",", type=str, help="The csv delimiter")
        # parser.add_argument('--quotechar', '-q', default='"', type=str,
        #                     help="The character used for quoting in the csv")
        parser.add_argument('--custom_class', '-e', type=str, default="JSONDataEntry", help="The custom DataEntry class to use")

        parser.add_argument('--drop_columns', '-c', default=tuple(), nargs="*", type=str,
                            help="The dummy_columns that need to be dropped")
        parser.add_argument('--time_column', '-t', type=str, help="The column used as timestamps")
        parser.add_argument('--datetime_format', '-f', default=now(), type=str, help="The format of the time_column")
        parser.add_argument('--dataframe', '-d', type=str, help="The name of the dataframe", required=True)
        parser.add_argument('path', nargs='+', type=str, help="Path to the csv('s)")

    def handle(self, *args, **options):
        for path in options["path"]:
            df = pandas.read_csv(path)
            if len(options["drop_columns"]) > 0:
                df = df.drop(options["drop_columns"], axis=1)
            dfm = DataframeModel.objects.create(name=options["dataframe"])
            for index, column in enumerate(df.columns):
                DataframeColumn.objects.create(name=column, dataframe=dfm, index=index)

            DataEntryClass = find_subclass_by_name(options["custom_class"], BaseDataEntry)
            for index, row in df.iterrows():
                DataEntryClass(
                    dataframe=dfm,
                    created_on=options["time_column"],
                    datetime_format=options["datetime_format"],
                    **(row.to_dict())
                ).save()

