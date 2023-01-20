from django.urls import path,re_path
from django.conf.urls import url
from training import views

urlpatterns = [

               url(r'^deploy_experiment/$', views.deploy_experiment),
               url(r'^create_experiment/$', views.create_experiment),
               url(r'^get_deployment_list/$', views.get_deployment_list),
               url(r'^get_all_running_experiments/$', views.get_all_running_experiments), 
               #url(r'^get_running_experiment/(?P<part_id>[A-Za-z0-9-_]+)$', views.get_running_experiment), 
               url(r'^experiment_status/(?P<part_id>[A-Za-z0-9-_]+)/(?P<experiment_id>[A-Za-z0-9-_]+)$', views.get_experiment_status),
               #url(r'^deploy_experiment/$', views.deploy_experiment),
               url(r'^get_trained_list/$', views.get_trained_list),


               ###monk objdet t2###
               #url(r'^monk_list_models/$', views.monk_list_models),
               #url(r'^run_monk_train/$', views.run_monk_train),
               

]
