from django.contrib.auth.models import User, Group
from rest_framework import serializers
from server.models.models import RegResults


class ResSerializer(serializers.HyperlinkedModelSerializer):
    class Meta:
        model = RegResults
        fields = ('url', 'value', 'realValue', 'absoluteError', 'relativeError', 'timeConsumption')
