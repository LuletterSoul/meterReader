from django.db import models
from ..settings import IMAGE_DIR, TEMPLATE_DIR, CONFIG_DIR, PROC_REL_DIR
from ..settings import IMAGE_REL_DIR, TEMPLATE_REL_DIR, CONFIG_REL_DIR
import os


# Create your models here.
class Template(models.Model):
    template = models.ImageField(upload_to=TEMPLATE_REL_DIR, default=os.path.join(TEMPLATE_REL_DIR, 'default.jpg'))
    # uuid = models.UUIDField(u'uuid', default=uuid.uuid4())


class Config(models.Model):
    config = models.FileField(upload_to=CONFIG_REL_DIR, default=os.path.join(CONFIG_REL_DIR, 'default.json'))
    # uuid = models.UUIDField(u'uuid', default=uuid.uuid4())


class ImageSrc(models.Model):
    src = models.ImageField(upload_to=IMAGE_REL_DIR, default=os.path.join(IMAGE_REL_DIR, 'default.jpg'))
    # result = models.ForeignKey('RegResult', on_delete=models.DO_NOTHING)
    # uuid = models.UUIDField(u'uuid', default=uuid.uuid4())


class Proc(models.Model):
    # src = models.ForeignKey('RegResult', on_delete=models.DO_NOTHING)
    proc = models.ImageField(upload_to=PROC_REL_DIR, default=os.path.join(PROC_REL_DIR, 'default.jpg'))
    result = models.ForeignKey('RegResult',on_delete=models.DO_NOTHING)


# Create your models here.
class RegResult(models.Model):
    # id = models.IntegerField(primary_key=True,)
    # src = models.ImageField(upload_to=IMAGE_REL_DIR)
    # src = models.ForeignKey(ImageSrc, on_delete=models.DO_NOTHING)
    value = models.FloatField(u'识别值', default=-1)
    realValue = models.FloatField(u'真值', default=-1)
    absoluteError = models.FloatField(u'绝对误差', default=-1)
    relativeError = models.FloatField(u'相对误差', default=-1)
    timeConsumption = models.FloatField(u'耗时', default=0)
    # proc = models.ManyToManyField(Proc)
    # src = models.ImageField(upload_to=IMAGE_REL_DIR, default=os.path.join(IMAGE_REL_DIR, 'default.jpg'))
    # template = models.ImageField(upload_to=TEMPLATE_REL_DIR, default=os.path.join(TEMPLATE_REL_DIR, 'default.jpg'))
    # config = models.FileField(upload_to=CONFIG_REL_DIR, default=os.path.join(CONFIG_REL_DIR, 'default.json'))
    srcId = models.IntegerField(u'源图像id', default=0)
    # src = models.ForeignKey(ImageSrc, on_delete=models.DO_NOTHING, null=True)
    # template = models.ForeignKey(Template, on_delete=models.DO_NOTHING)
    # config = models.ForeignKey(Config, on_delete=models.DO_NOTHING)
