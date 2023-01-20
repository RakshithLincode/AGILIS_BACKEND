from django.urls import path,re_path
from parts import views


urlpatterns = [
    re_path(r'^add_part/$', views.add_part_details),
    re_path(r'^delete_part/(?P<part_id>[A-Za-z0-9-_]+)$', views.delete_part),
    re_path(r'^update_part/$', views.update_part),
    re_path(r'^part_details/(?P<part_id>[A-Za-z0-9-_]+)', views.get_part_details),
    re_path(r'^get_all_parts/$', views.get_parts),
    re_path(r'^get_partInfo/(?P<short_number>[A-Za-z0-9-_]+)', views.get_partInfo),
    re_path(r'^get_short_numbers_list/$', views.get_short_numbers_list),

]
