from django.db import models


class RegResults(models.Model):
    value = models.FloatField
    realValue = models.FloatField
    absoluteError = models.FloatField
    relativeError = models.FloatField
    timeConsumption = models.FloatField

