from rest_framework import viewsets
from rest_framework.parsers import MultiPartParser, FormParser
from django_filters.rest_framework import DjangoFilterBackend
from rest_framework import filters
from django_filters import rest_framework
from rest_framework.settings import api_settings
from rest_framework import status
from rest_framework.response import Response
from ..settings import PROC_DIR, PROC_MAIN_DIR
from .PointerMeterReg2 import entry
from .filters import *

from django.core import serializers
from dss.Serializer import serializer
# from Algorithm import *
from .serializers import *

MEDIA_URL = '/media/'
MEDIA_ROOT = os.path.join('', 'media')


# Create your views here.
class ResViewSet(viewsets.ModelViewSet):
    """
    API endpoint that allows users to be viewed or edited.
    """
    queryset = RegResult.objects.all()
    serializer_class = ResSerializer

    def create(self, request, *args, **kwargs):
        # self.perform_create(serializer)
        # print(serializer)
        data = request.data
        src_id = data['srcId']
        img = ImageSrc.objects.get(id=data['srcId'])
        if img is None:
            print("Image is None.")
        result = entry(src_id, os.path.basename(img.src.name), CONFIG_DIR, img.src.path, TEMPLATE_DIR, PROC_MAIN_DIR)
        print(result)
        jsonSerializer = ResSerializer(result)
        return Response(jsonSerializer.data, status=status.HTTP_201_CREATED)

    def perform_create(self, serializer):
        serializer.save()

    def get_success_headers(self, data):
        try:
            return {'Location': str(data[api_settings.URL_FIELD_NAME])}
        except (TypeError, KeyError):
            return {}


class UploadImageSrcView(viewsets.ModelViewSet):
    parser_classes = (MultiPartParser, FormParser)
    queryset = ImageSrc.objects.all()
    serializer_class = ImgSrcSerializer

    # def post(self, request, format=None):
    #     files = request.FILES.getlist('files')
    #     if len(files) > 0:
    #         if not os.path.exists(MEDIA_ROOT):
    #             os.makedirs(MEDIA_ROOT)
    #             for file in files:
    #                 file_name, extention = os.path.splitext(file.name)
    #                 file_path = '{}/{}'.format(MEDIA_ROOT, file_name)
    #                 with open(file_path, 'wb') as f:
    #                     for c in file.chuncks():
    #                         f.write(c)
    #         return 'success'
    #     else:
    #         return 'failed'
    #
    # def create(self, request, *args, **kwargs):
    #     # serializer = self.get_serializer(data=request.data)
    #     # serializer.is_valid(raise_exception=True)
    #     # self.perform_create(serializer)
    #     # headers = self.get_success_headers(serializer.data)
    #     print(request)
    #     files = request.FILES.getlist('files')
    #     if len(files) > 0:
    #         if not os.path.exists(MEDIA_ROOT):
    #             os.makedirs(MEDIA_ROOT)
    #             for file in files:
    #                 file_name, extention = os.path.splitext(file.name)
    #                 file_path = '{}/{}'.format(MEDIA_ROOT, file_name)
    #                 with open(file_path, 'wb') as f:
    #                     for c in file.chuncks():
    #                         f.write(c)
    #         return 'success'
    #     else:
    #         return 'failed'
    #     return Response(serializer.data, status=status.HTTP_201_CREATED, headers=headers)
    #
    # def perform_create(self, serializer):
    #     serializer.save()
    #
    # def get_success_headers(self, data):
    #     try:
    #         return {'Location': str(data[api_settings.URL_FIELD_NAME])}
    #     except (TypeError, KeyError):
    #         return {}


class UploadTemplateView(viewsets.ModelViewSet):
    parser_classes = (MultiPartParser, FormParser)
    queryset = Template.objects.all()
    serializer_class = TemplateSerializer


class UploadConfigView(viewsets.ModelViewSet):
    parser_classes = (MultiPartParser, FormParser)
    queryset = Config.objects.all()
    serializer_class = ConfigSerializer


class ProcView(viewsets.ModelViewSet):
    # filter_backends = (rest_framework.DjangoFilterBackend, filters.SearchFilter, filters.OrderingFilter)
    # filter_class = ProcFilter
    queryset = Proc.objects.all()
    serializer_class = ProcSerializer
    # def get(self, request, *args, **kwargs):

