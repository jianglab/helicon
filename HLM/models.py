from django.db import models
from django.contrib.auth.models import User

class Data(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    created_at = models.DateTimeField(auto_now_add=True)
    file = models.FileField(upload_to='data_files/')

    def __str__(self):
        return f'{self.user.username} - {self.created_at}'
