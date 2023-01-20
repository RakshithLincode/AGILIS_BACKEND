from django.urls import path,re_path
from toyoda import views

urlpatterns = [
    re_path(r'^start_process/$', views.start_process_toyoda),
    re_path(r'^end_process/$', views.end_process_toyoda),
    re_path(r'^get_running_process/(?P<workstation_id>[A-Za-z0-9-_]+)', views.get_toyoda_running_process),
    re_path(r'^update_manual_qc/$', views.update_manual_inspection_result),
    re_path(r'^getCameraFeeds/(?P<workstation_id>[A-Za-z0-9-_]+)', views.get_camera_feed_urls),
    re_path(r'^plan_production_counter_modify/$', views.plan_production_counter_modify),
    re_path(r'^rescan/$', views.rescan),
    re_path(r'^generate_QRcode/(?P<inspection_id>[A-Za-z0-9-_]+)', views.generate_QRcode),
    re_path(r'^stream/(?P<wid>[A-Za-z0-9-_]+)/(?P<cameraid>[A-Za-z0-9-_.]+)/$', views.get_camera_stream), 
    re_path(r'^stream1/(?P<key>[A-Za-z0-9-_.]+)/$', views.get_redis_stream), 
    re_path(r'^get_inspection_qc_list/(?P<process_id>[A-Za-z0-9-_]+)', views.get_inspection_qc_list),
]
