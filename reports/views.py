from django.shortcuts import render
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from rest_framework.decorators import api_view, renderer_classes
from rest_framework.renderers import TemplateHTMLRenderer,JSONRenderer
from django.http import HttpResponse
import json
from common.utils import *

from drf_yasg import openapi
from drf_yasg.openapi import Schema, TYPE_OBJECT, TYPE_STRING, TYPE_ARRAY
from drf_yasg.utils import swagger_auto_schema


#nss
@api_view(['POST'])
@csrf_exempt
#@check_group(['admin'])
def detail_report_util(request):
    #request = json.loads(request.body)
    from reports.utils import detail_report_util
    p = json.loads(request.body)
    response = detail_report_util(p)
    print(p,'fggggggggggggggggggggggggggggggggg')
    return HttpResponse(json.dumps(response, cls=Encoder), content_type="application/json")

@api_view(['GET'])
@csrf_exempt
#@check_group(['admin'])
def get_operator_list(request):
    from reports.utils import operator_list
    response = operator_list()
    return HttpResponse(json.dumps(response, cls=Encoder), content_type="application/json")

@api_view(['POST'])
@renderer_classes((TemplateHTMLRenderer,))
@csrf_exempt
#@check_group(['admin'])
def detail_report_export_util(request):
    from reports.utils import detail_report_export_util
    p = json.loads(request.body)
    # data = json.loads(request.body)
    print(p,'dddsanijanihcbiacibduicbsudhbcvisdh')
    resp = detail_report_export_util(p)
    return HttpResponse(json.dumps(resp, cls=Encoder), content_type="application/json")




@api_view(['GET'])
@renderer_classes((TemplateHTMLRenderer,JSONRenderer))
@csrf_exempt
def get_metrics(request, inspectionid):
    from reports.utils import get_metrics_util
    resp = get_metrics_util(inspectionid)
    return  HttpResponse(json.dumps(resp, cls=Encoder), content_type="application/json")


@api_view(['GET'])
@renderer_classes((TemplateHTMLRenderer,JSONRenderer))
@csrf_exempt
def get_last_defect_list(request, inspectionid):
    from reports.utils import get_last_defect_list_util
    skip = request.GET.get('skip', 0)
    limit = request.GET.get('limit', 20)
    inspection_id = request.GET.get('')
    resp = get_last_defect_list_util(inspectionid ,skip, limit)
    print(resp)
    return HttpResponse(json.dumps(resp, cls=Encoder), content_type="application/json")


@api_view(['POST'])
@renderer_classes((TemplateHTMLRenderer,JSONRenderer))
@csrf_exempt
def get_accepted_rejected_parts_list(request):
    from reports.utils import get_accepted_rejected_parts_list_util
    p = json.dumps(request.POST)
    p = json.loads(p)
    start_date = p['start_date'] if 'start_date' in p else None
    end_date = p['end_date'] if 'end_date' in p else None
    status = p['status'] if 'status' in p else None
    response = get_accepted_rejected_parts_list_util(start_date, end_date, status)
    return HttpResponse(json.dumps(response, cls=Encoder), content_type="application/json")

    
@swagger_auto_schema(method='post', request_body=openapi.Schema(
    type=openapi.TYPE_OBJECT, 
    properties={
        'from_date' : openapi.Schema(type=openapi.TYPE_STRING, example='25-08-2020 13:59:04'),
        'to_date' : openapi.Schema(type=openapi.TYPE_STRING, example='25-08-2020 23:59:04'),
        'operator_name': openapi.Schema(type=openapi.TYPE_STRING, example='prod2 supervisor2'),
        'defect_type' : openapi.Schema(type=openapi.TYPE_STRING, example=[]),
        'feature_type': openapi.Schema(type=openapi.TYPE_STRING, example=["Part_Presence", "PPWSRSLH", "Felt_Presence"])
    }
))
@api_view(['POST'])
@renderer_classes((TemplateHTMLRenderer,JSONRenderer))
@csrf_exempt
def get_defect_list_report(request):
    from reports.utils import get_defect_list_report_util
    data = json.loads(request.body)
    response = get_defect_list_report_util(data)
    return HttpResponse(json.dumps(response, cls=Encoder), content_type="application/json")


@api_view(['GET'])
@renderer_classes((TemplateHTMLRenderer,JSONRenderer))
@csrf_exempt
def get_summary_end_process(request,inspectionid):
    from reports.utils import get_summary_end_process_util
    response = get_summary_end_process_util(inspectionid)
    return HttpResponse(json.dumps(response, cls=Encoder), content_type="application/json")


@api_view(['POST'])
@renderer_classes((TemplateHTMLRenderer,JSONRenderer))
@csrf_exempt
def defect_type_based_report(request):
    from reports.utils import defect_type_based_report_util
    data = json.loads(request.body)
    response = defect_type_based_report_util(data)
    return HttpResponse(json.dumps(response, cls=Encoder), content_type="application/json")


@api_view(['GET'])
@csrf_exempt
def get_master_defect_list(request):
    from reports.utils import get_master_defects
    resp = get_master_defects()
    return HttpResponse(json.dumps(resp, cls=Encoder), content_type="application/json")


@api_view(['GET'])
@csrf_exempt
def get_master_feature_list(request):
    from reports.utils import get_master_features
    resp = get_master_features()
    return HttpResponse(json.dumps(resp, cls=Encoder), content_type="application/json")
