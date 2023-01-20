from django.urls import path,re_path
from reports import views

urlpatterns = [
    re_path(r'^get_metrics/(?P<inspectionid>[A-Za-z0-9-_]+)', views.get_metrics),
    re_path(r'^get_defect_list/(?P<inspectionid>[A-Za-z0-9-_]+)', views.get_last_defect_list),
    re_path(r'^get_accepted_rejected_parts/$', views.get_accepted_rejected_parts_list),
    re_path(r'^getDefectListReport/$', views.get_defect_list_report),
    re_path(r'^getSummary/(?P<inspectionid>[A-Za-z0-9-_]+)', views.get_summary_end_process),
    re_path(r'^getDefectTypeBasedReport/$', views.defect_type_based_report),
    re_path(r'^get_master_defect_list/$', views.get_master_defect_list),
    re_path(r'^get_master_feature_list/$', views.get_master_feature_list),
    
    

    ### nss
    re_path(r'^detail_report_util/$', views.detail_report_util),
    re_path(r'^get_operators/$', views.get_operator_list),
    re_path(r'^detail_report_export_util/$', views.detail_report_export_util)
]


# /livis/v1/reports/detail_report_util/