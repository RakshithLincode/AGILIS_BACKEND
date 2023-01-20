from django.contrib import admin
from accounts.models import User,User_Client,User_SI,Client,SI

admin.site.register(User)
admin.site.register(User_Client)
admin.site.register(User_SI)
admin.site.register(Client)
admin.site.register(SI)
