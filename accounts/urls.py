from django.urls import path,re_path
from accounts import views

urlpatterns = [
    re_path(r'^add_user_account/$', views.add_user_account),
    re_path(r'^get_user_account/(?P<client_id>[A-Za-z0-9-_]+)', views.get_user_account),
    re_path(r'^update_user_account/$', views.update_user_account),
    re_path(r'^delete_user_account/(?P<client_id>[A-Za-z0-9-_]+)$', views.delete_user_account),
    re_path(r'^user_accounts/$', views.get_all_user_accounts),

    re_path(r'^add_user_client/$', views.add_user_client),
    re_path(r'^get_user_client/(?P<client_id>[A-Za-z0-9-_]+)', views.get_user_client),
    re_path(r'^update_user_client/$', views.update_user_client),
    re_path(r'^delete_user_client/(?P<client_id>[A-Za-z0-9-_]+)$', views.delete_user_client),
    re_path(r'^get_user_clients/$', views.get_user_clients),

    re_path(r'^add_user_si/$', views.add_user_si),
    re_path(r'^get_user_si/(?P<client_id>[A-Za-z0-9-_]+)', views.get_user_si),
    re_path(r'^update_user_si/$', views.update_user_si),
    re_path(r'^delete_user_si/(?P<client_id>[A-Za-z0-9-_]+)$', views.delete_user_si),
    re_path(r'^get_user_sis/$', views.get_user_sis),

    re_path(r'^add_client_account/$', views.add_client_account),
    re_path(r'^get_client_account/(?P<client_id>[A-Za-z0-9-_]+)', views.get_client_account),
    re_path(r'^update_client_account/$', views.update_client_account),
    re_path(r'^delete_client_account/(?P<client_id>[A-Za-z0-9-_]+)$', views.delete_client_account),
    re_path(r'^client_accounts/$', views.get_all_client_accounts),

    re_path(r'^add_si_account/$', views.add_si_account),
    re_path(r'^get_si_account/(?P<client_id>[A-Za-z0-9-_]+)', views.get_si_account),
    re_path(r'^update_si_account/$', views.update_si_account),
    re_path(r'^delete_si_account/(?P<client_id>[A-Za-z0-9-_]+)$', views.delete_si_account),
    re_path(r'^si_accounts/$', views.get_all_si_accounts),

    re_path(r'^login/$', views.login_user),
    re_path(r'^login_supervisor/$', views.login_supervisor),
    
    re_path(r'^change_password/$', views.change_password),

    ]
