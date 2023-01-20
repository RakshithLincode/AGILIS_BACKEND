
from django.urls import path, re_path

from annotate import views


urlpatterns = [
                       #manual annotation apis
                       re_path(r'^next_img/$', views.next_img),
                       re_path(r'^prev_img/$', views.prev_img),
                       re_path(r'^get_img/$', views.get_img),
                       re_path(r'^delete_img/$', views.delete_img),
                       
                       re_path(r'^get_dataset/$', views.get_dataset_list),
                       
                       #re_path(r'^create_dataset/$', views.create_dataset),
                       re_path(r'^get_data_for_histogram/$', views.get_data_for_histogram),
                       #re_path(r'^card_flip_random_image/$', views.card_flip_random_image),
                       re_path(r'^fetch_data/$', views.fetch_data),
                       re_path(r'^submit_annotations/$', views.submit_annotations),
                       #re_path(r'^check_annotations/$', views.check_annotations),
                       re_path(r'^export_data/$', views.export_data),
                       re_path(r'^fetch_image_url/$', views.fetch_image_url),

                       ##auto-annotation apis
                       #re_path(r'^auto_salient_annotations/$', views.auto_salient_annotations),
                       #re_path(r'^auto_color_annotations/$', views.auto_color_annotations),

                       ## Capture Urls
                       #re_path(r'^get_capture_feed_url/$', views.get_capture_feed_url),
                       #re_path(r'^capture_part_image/$', views.capture_part_image),
                       #re_path(r'^start_capture_camera/$', views.start_camera),
                       #re_path(r'^stop_capture_camera/$', views.stop_camera)
                       
]
