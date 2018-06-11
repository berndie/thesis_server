from django.contrib.auth.models import User
from django.db import models
from django.db.models import TextField


class PredictionSubject(models.Model):
    identifier = TextField()

class FirebaseUser(PredictionSubject):
    firebase_token = TextField()