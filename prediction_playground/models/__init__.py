from datetime import datetime

from dateutil import parser
from django.db import models
from django.utils import timezone


class DatetimeParser(object):
    @staticmethod
    def parse_time(string, datetime_format=None):
        if string is None:
            time = datetime.now()
        elif isinstance(string, datetime):
            time = string
        elif datetime_format is None:
            time = parser.parse(string)
        else:
            time = datetime.strptime(string, datetime_format)
        if time.tzinfo is None:
            time = time.replace(tzinfo=timezone.get_current_timezone())
        return time


class BetterDateTimeField(models.DateTimeField):
    def get_prep_value(self, value):
        value = DatetimeParser.parse_time(value)
        return super(DateTimeField, self).get_prep_value(value)


from users import *
from models import *
from managers import *
