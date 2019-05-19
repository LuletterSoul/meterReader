import django_filters

from .models import *

#
# class ProcFilter(django_filters.rest_framework.FilterSet):
#     """
#     物理服务器过滤器
#     """
#
#     id = django_filters.NumberFilter(name='id',)
#
#     class Meta:
#         model = Proc
#         fields = ['id', 'proc']
