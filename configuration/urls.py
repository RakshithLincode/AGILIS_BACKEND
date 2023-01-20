
from django.urls import path, re_path

from configuration import views


urlpatterns = [
                       #kanban component configuration apis
                       re_path(r'^fetch_individual_component_list/$', views.fetch_individual_component_list),
                       #re_path(r'^fetch_individual_component_list/(?P<jig_id>[A-Za-z0-9-_]+)', views.fetch_individual_component_list),
                       re_path(r'^fetch_jig_list/$', views.fetch_jig_list),
                       re_path(r'^add_jig/$', views.add_jig),
                       re_path(r'^fetch_specific_jig/(?P<jig_id>[A-Za-z0-9-_]+)$', views.fetch_specific_jig),
                       re_path(r'^update_jig/$', views.update_jig),
                       re_path(r'^delete_jig/$', views.delete_jig),
                       re_path(r'^list_specific_jig/(?P<jig_type>[A-Za-z0-9-_]+)$', views.list_specific_jig)
]
