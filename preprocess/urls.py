
from django.urls import path, re_path

from preprocess import views


urlpatterns = [

                       re_path(r'^set_crop/$', views.set_crop),



                       ## Capture Urls
                       re_path(r'^get_capture_feed_url/$', views.get_capture_feed_url),
                       re_path(r'^stream/(?P<wid>[A-Za-z0-9-_]+)/(?P<cameraid>[A-Za-z0-9-_.]+)/$', views.get_camera_stream), 
                       re_path(r'^stream1/(?P<key>[A-Za-z0-9-_.]+)/$', views.get_redis_stream), 
                       re_path(r'^initial_capture/$', views.initial_capture),
                       re_path(r'^show_captured_img/$', views.show_captured_img),
                       re_path(r'^set_init_regions/$', views.set_init_regions),
                       re_path(r'^capture_util/$', views.capture_util),
                       re_path(r'^change_img/$', views.change_img),
                       re_path(r'^final_capture/$', views.final_capture),
                       #re_path(r'^capture_part_image/$', views.capture_part_image),
                       #re_path(r'^start_capture_camera/$', views.start_camera),
                       #re_path(r'^stop_capture_camera/$', views.stop_camera)
                       
]
