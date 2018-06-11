from django.conf.urls import url

from forever_alone.views import FAReceiveView

urlpatterns = [
    url(r'^api/', FAReceiveView.as_view(), name='fa_receive'),
]