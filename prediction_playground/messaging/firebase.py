import json
import os
import sys

import firebase_admin
from django.conf import settings
from firebase_admin import credentials, messaging

from prediction_playground.messaging import BaseMessager, BaseMessage


class FirebaseMessager(BaseMessager):

    def __init__(self, cert_path=settings.PREDICTION_PLAYGROUND_FIREBASE_CERT_LOCATION):
        try:
            firebase_admin.get_app()
        except ValueError:
            cred = credentials.Certificate(cert_path)
            firebase_admin.initialize_app(cred)

    def send_message(self, message):
        fb_message = messaging.Message(
            notification=messaging.Notification(title=message.title, body=message.body),
            # aware, data can only by dictonaries with string values
            data=message.payload,
            token=message.user.firebase_token,
        )

        # Send a message to the device corresponding to the provided
        # registration token.
        dry_run = settings.TESTING
        return messaging.send(fb_message, dry_run=dry_run)

    def send_prediction(self, prediction, user):
        """Send the predictions to their owners"""
        msg = BaseMessage(
            user,
            "Your prediction is ready",
            "",
            {"predictions": json.dumps(list(prediction))}
        )
        self.send_message(msg)
