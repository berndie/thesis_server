from django.conf.urls import url, include
from django.contrib import admin

from prediction_playground.views import OverviewView, FirebaseReceiveView,  \
    JSONReceiveView

version = 1


api_patterns = [

]
urlpatterns = [
    url(r'^api/v%s/' % str(version), include(api_patterns)),
    url(r'^json_receive/', JSONReceiveView.as_view(), name='json_receive'),
    url(r'^overview/(?P<pipeline_name>[\w-]+)', OverviewView.as_view(), name="overview"),
]