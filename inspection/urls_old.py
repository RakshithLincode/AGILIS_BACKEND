
from django.urls import path, re_path

from inspection import views


urlpatterns = [
                       #operator api
                       re_path(r'^get_current_inspection_details/(?P<jig_id>[A-Za-z0-9-_]+)$', views.get_current_inspection_details),
                       re_path(r'^start_inspection/$', views.start_process_schneider),
                       re_path(r'^force_admin_pass/$', views.force_admin_pass), #when admin force pass it
                       re_path(r'^get_running_process/$', views.get_running_process), #when he press refresh
]
