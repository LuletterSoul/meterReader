from rest_framework import viewsets
from rest_framework.parsers import MultiPartParser, FormParser, FileUploadParser
import hashlib
import json
from django.core.files import File
from ..settings.base import *
from django_filters.rest_framework import DjangoFilterBackend
from rest_framework import filters
from django_filters import rest_framework
from rest_framework.settings import api_settings
from rest_framework import status
from rest_framework.response import Response
from django.http.response import JsonResponse
from ..settings.base import PROC_DIR, PROC_MAIN_DIR
from .PointerMeterReg2 import entry
from .filters import *
import threading

from django.core import serializers
from dss.Serializer import serializer
# from Algorithm import *
from .serializers import *


# MEDIA_URL = '/media/'
# MEDIA_ROOT = os.path.join('', 'media')
class MyThread(threading.Thread):

    def __init__(self, func, args=()):
        super(MyThread, self).__init__()
        self.func = func
        self.args = args

    def run(self):
        self.result = self.func(*self.args)

    def get_result(self):
        try:
            return self.result  # 如果子线程不使用join方法，此处可能会报没有self.result的错误
        except Exception:
            return None


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
        src_ids = data['srcIds']
        results = []
        tasks = []
        if src_ids is not None:
            datas = ImageSrc.objects.filter(id__in=src_ids)
            for index, d in enumerate(datas):
                path = os.path.join(BASE_DIR, d.src)
                if os.path.exists(path):
                    with open(d.src) as img:
                        if img is None:
                            print("Image is None.")
                        t = MyThread(entry, args=(
                            src_ids[index], os.path.basename(img.name), CONFIG_DIR, img.name, TEMPLATE_DIR,
                            PROC_DIR))
                    t.start()
                    tasks.append(t)
                else:
                    print('Image not existed.')
            for t in tasks:
                t.join()
                result = t.get_result()
                if result is not None:
                    results.append(result)
                    results.append(result)
            jsonSerializer = ResSerializer(results, many=True)
        elif src_id is not None:
            data = ImageSrc.objects.get(id=data['srcId'])
            path = os.path.join(BASE_DIR, data.src)
            if os.path.exists(path):
                with open(data.src) as img:
                    if img is None:
                        print("Image is None.")
                    result = entry(src_id, os.path.basename(img.name), CONFIG_DIR, img.name, TEMPLATE_DIR, PROC_DIR)
                    jsonSerializer = ResSerializer(result, many=False)
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
    def create(self, request, *args, **kwargs):
        file = request.data['file']
        # if len(files) > 0:
        if not os.path.exists(IMAGE_DIR):
            os.makedirs(IMAGE_DIR)
            # for file in files:
        filename = file.name.lower()
        file_path = '{}/{}'.format(IMAGE_DIR, filename)
        media_path = '{}/{}'.format(IMAGE_REL_DIR, filename)
        srcImg = ImageSrc()
        # 重复返回第一个文件
        if os.path.exists(media_path):
            filter = ImageSrc.objects.filter(src=media_path)
            if len(filter):
                srcImg = filter[0]
        # 不重复，写入文件，保存到数据库
        else:
            with open(file_path, 'wb') as f:
                for c in file.chunks():
                    f.write(c)
            srcImg = ImageSrc(src=media_path, filename=filename)
            srcImg.save()
        return Response(ImgSrcSerializer(srcImg).data, status=status.HTTP_201_CREATED)


class UploadTemplateView(viewsets.ModelViewSet):
    parser_classes = (MultiPartParser, FormParser)
    queryset = Template.objects.all()
    serializer_class = TemplateSerializer

    def create(self, request, *args, **kwargs):
        file = request.data['file']
        # if len(files) > 0:
        if not os.path.exists(TEMPLATE_DIR):
            os.makedirs(TEMPLATE_DIR)
            # for file in files:
        filename = file.name.lower()
        file_path = '{}/{}'.format(TEMPLATE_DIR, filename)
        media_path = '{}/{}'.format(TEMPLATE_REL_DIR, filename)
        with open(file_path, 'wb') as f:
            for c in file.chunks():
                f.write(c)
        t = Template(template=media_path, filename=filename)
        t.save()
        # ImageSrc.objects.create(src=media_path)
        # created.append(src)
        # json_data = serializers.serialize('json', created)
        # json_data = serializer(created, )
        # return JsonResponse(json.dumps(created), status=status.HTTP_201_CREATED)
        return Response(TemplateSerializer(t).data, status=status.HTTP_201_CREATED)


class UploadConfigView(viewsets.ModelViewSet):
    parser_classes = (MultiPartParser, FormParser)
    queryset = Config.objects.all()
    serializer_class = ConfigSerializer

    def create(self, request, *args, **kwargs):
        file = request.data['file']
        # if len(files) > 0:
        if not os.path.exists(CONFIG_DIR):
            os.makedirs(CONFIG_DIR)
            # for file in files:
        filename = file.name.lower()
        file_path = '{}/{}'.format(CONFIG_DIR, filename)
        media_path = '{}/{}'.format(CONFIG_REL_DIR, filename)
        with open(file_path, 'wb') as f:
            for c in file.chunks():
                f.write(c)
        c = Config(config=media_path, filename=filename)
        c.save()
        # ImageSrc.objects.create(src=media_path)
        # created.append(src)
        # json_data = serializers.serialize('json', created)
        # json_data = serializer(created, )
        # return JsonResponse(json.dumps(created), status=status.HTTP_201_CREATED)
        return Response(ConfigSerializer(c).data, status=status.HTTP_201_CREATED)


class ProcView(viewsets.ModelViewSet):
    # filter_backends = (rest_framework.DjangoFilterBackend, filters.SearchFilter, filters.OrderingFilter)
    # filter_class = ProcFilter
    queryset = Proc.objects.all()
    serializer_class = ProcSerializer

    def list(self, request, *args, **kwargs):
        res_id = request.GET.get("resultId")
        if res_id:
            return Response(
                ProcSerializer(
                    Proc.objects.filter(result=RegResult.objects.filter(id=int(res_id))[0]).order_by('order'),
                    many=True).data)
        else:
            return Response(ProcSerializer(Proc.objects.all(), many=True).data)
