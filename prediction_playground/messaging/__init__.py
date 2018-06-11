class BaseMessager(object):
    """Interface for messagers to send feedback to a user"""
    def send_message(self, message):
        """Method to send a message to a user

        :param message: The message
        :type message: BaseMessage
        :return:
        """
        raise NotImplementedError("You must override this method")

    def send_prediction(self, prediction, user):
        raise NotImplementedError("You must override this method")




class BaseMessage(object):
    """An abstract message class"""
    def __init__(self, send_to, title, body, payload):
        """

        :param send_to: User object to send to
        :param title: title of the message
        :param body: body of the message
        :param payload: extra payload for the message
        """
        self.user = send_to
        self.title = title
        self.body = body
        self.payload = payload
