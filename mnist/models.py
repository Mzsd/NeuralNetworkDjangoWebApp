from django.db import models

class Img(models.Model):
    image = models.URLField()
    label = models.TextField(default="")