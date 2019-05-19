from rest_framework import serializers
from .models import *


class ResSerializer(serializers.HyperlinkedModelSerializer):
    class Meta:
        model = RegResult
        fields = ('id', 'value', 'srcId', 'realValue', 'absoluteError', 'relativeError', 'timeConsumption')
        # proc = serializers.HyperlinkedIdentityField(view_name='proc-detail', lookup_field='id')


class ImgSrcSerializer(serializers.HyperlinkedModelSerializer):
    class Meta:
        model = ImageSrc
        fields = ('id', 'src')


class ConfigSerializer(serializers.HyperlinkedModelSerializer):
    class Meta:
        model = Config
        fields = ('id', 'config')


class TemplateSerializer(serializers.HyperlinkedModelSerializer):
    class Meta:
        model = Template
        fields = ('id', 'template')


class ProcSerializer(serializers.HyperlinkedModelSerializer):
    class Meta:
        model = Proc
        fields = ('id', 'proc')
