from django.shortcuts import render
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from rest_framework.decorators import api_view, renderer_classes
from rest_framework.renderers import TemplateHTMLRenderer,JSONRenderer
from django.http import HttpResponse,StreamingHttpResponse
import json
from common.utils import *

from drf_yasg import openapi
from drf_yasg.openapi import Schema, TYPE_OBJECT, TYPE_STRING, TYPE_ARRAY
from drf_yasg.utils import swagger_auto_schema


@swagger_auto_schema(method='post', request_body=openapi.Schema(
    type=openapi.TYPE_OBJECT, 
    properties={
        'workstation_id' : openapi.Schema(type=openapi.TYPE_STRING, example='5f080781e987304f98e77d38'),
        'user_id' : openapi.Schema(type=openapi.TYPE_STRING, example='5ef5005a-5473-440b-ab29-272a810da5d3'),
        'part_number': openapi.Schema(type=openapi.TYPE_STRING, example='5f2d511274fcc0e8c19a6946'),
        'model_number' : openapi.Schema(type=openapi.TYPE_STRING, example='md2'),
        'short_number': openapi.Schema(type=openapi.TYPE_STRING, example='IG98'),
        'part_description' : openapi.Schema(type=openapi.TYPE_STRING, example='pdbdlndl'),
        'current_production_count' : openapi.Schema(type=openapi.TYPE_INTEGER, example=200)
    }
))
@api_view(['POST'])
@renderer_classes((TemplateHTMLRenderer,JSONRenderer))
@csrf_exempt
def start_process_toyoda(request):
    data = json.loads(request.body)
    from toyoda.utils import start_toyoda_process
    resp = start_toyoda_process(data)
    return HttpResponse(json.dumps(resp, cls=Encoder), content_type="application/json") 


@swagger_auto_schema(method='post', request_body=openapi.Schema(
    type=openapi.TYPE_OBJECT, 
    properties={
        'process_id' : openapi.Schema(type=openapi.TYPE_STRING, example='5f2d942673efa9aaf736b186')
    }
))
@api_view(['POST'])
@renderer_classes((TemplateHTMLRenderer,JSONRenderer))
@csrf_exempt
def end_process_toyoda(request):
    data = json.loads(request.body)
    from toyoda.utils import end_toyoda_process
    response = end_toyoda_process(data)
    return HttpResponse(json.dumps(response,cls=Encoder), content_type="application/json") 


@api_view(['GET'])
@renderer_classes((TemplateHTMLRenderer,JSONRenderer))
@csrf_exempt
def get_toyoda_running_process(request,workstation_id):
    from toyoda.utils import get_toyoda_running_process
    response = get_toyoda_running_process(workstation_id)
    return HttpResponse(json.dumps(response,cls=Encoder), content_type="application/json") 


@swagger_auto_schema(method='post', request_body=openapi.Schema(
    type=openapi.TYPE_OBJECT, 
    properties={
        'qc_status' : openapi.Schema(type=openapi.TYPE_STRING, example='1'),
        'part_weight' : openapi.Schema(type=openapi.TYPE_STRING, example='ZE-376M1'),
        'qc_remark': openapi.Schema(type=openapi.TYPE_STRING, example='this is test1 remark by qc'),
        'inspection_type' : openapi.Schema(type=openapi.TYPE_STRING, example='l'),
        'process_id': openapi.Schema(type=openapi.TYPE_STRING, example='5f2d942673efa9aaf736b186')
    }
))
@api_view(['POST'])
@renderer_classes((TemplateHTMLRenderer,JSONRenderer))
@csrf_exempt
def update_manual_inspection_result(request):
    data = json.loads(request.body)
    from toyoda.utils import update_inspection_manual
    response = update_inspection_manual(data)
    return HttpResponse(json.dumps(response,cls=Encoder), content_type="application/json") 


@swagger_auto_schema(method='post', request_body=openapi.Schema(
    type=openapi.TYPE_OBJECT, 
    properties={
        'process_id' : openapi.Schema(type=openapi.TYPE_STRING, example='5f342b784d7295d5763ffd13'),
        'inspection_id' : openapi.Schema(type=openapi.TYPE_STRING, example='5f34cb691bed7473fc5f9196'),
        'serial_number': openapi.Schema(type=openapi.TYPE_INTEGER, example=1),
        'features_list' : openapi.Schema(type=openapi.TYPE_STRING, example=["clip", "felt"]),
        'defects_list': openapi.Schema(type=openapi.TYPE_STRING, example=['shot_shot_presence']),
        'taco_fail' : openapi.Schema(type=openapi.TYPE_STRING, example={"shot_shot_presence":1}),
        'taco_pass' : openapi.Schema(type=openapi.TYPE_STRING, example={"shot_shot_presence":1}),
        'is_accepted' : openapi.Schema(type=openapi.TYPE_BOOLEAN, example=False),
        'timestamp' : openapi.Schema(type=openapi.TYPE_STRING, example="2020-08-13 09:16:45")
    }
))
@api_view(['POST'])
@renderer_classes((TemplateHTMLRenderer,JSONRenderer))
@csrf_exempt
def rescan(request):
    data = json.loads(request.body)
    from toyoda.utils import rescan_util
    response = rescan_util(data)
    return HttpResponse(json.dumps(response,cls=Encoder), content_type="application/json") 


@api_view(['GET'])
@renderer_classes((TemplateHTMLRenderer,JSONRenderer))
@csrf_exempt
def get_camera_feed_urls(request, workstation_id):
    from toyoda.utils import get_camera_feed_urls
    resp = get_camera_feed_urls(workstation_id)
    return HttpResponse(json.dumps(resp, cls=Encoder), content_type="application/json")


@swagger_auto_schema(method='post', request_body=openapi.Schema(
    type=openapi.TYPE_OBJECT, 
    properties={
        'process_id' : openapi.Schema(type=openapi.TYPE_STRING, example='5f342b784d7295d5763ffd13'),
        'current_production_count' : openapi.Schema(type=openapi.TYPE_INTEGER, example=400),
    }
))
@api_view(['POST'])
@renderer_classes((TemplateHTMLRenderer,JSONRenderer))
@csrf_exempt
def plan_production_counter_modify(request):
    data = json.loads(request.body)
    from toyoda.utils import plan_production_counter_modify_util
    response = plan_production_counter_modify_util(data)
    return HttpResponse(json.dumps(response,cls=Encoder), content_type="application/json")


@api_view(['GET'])
@renderer_classes((TemplateHTMLRenderer,JSONRenderer))
@csrf_exempt
def generate_QRcode(request, inspection_id):
    from toyoda.utils import generate_QRcode_util
    response = generate_QRcode_util(inspection_id)
    return HttpResponse(json.dumps(response,cls=Encoder), content_type="application/json") 


@api_view(['GET'])
@csrf_exempt
def get_camera_stream(request, wid, cameraid):
    from toyoda.utils import redis_camera
    key = RedisKeyBuilderServer(wid).get_key(cameraid, 'predicted-frame')
    print(key)
    return StreamingHttpResponse(redis_camera(key), content_type='multipart/x-mixed-replace; boundary=frame')


@api_view(['GET'])
@csrf_exempt
def get_redis_stream(request, key):
    from toyoda.utils import redis_camera
    #key = RedisKeyBuilderServer(wid).get_key(cameraid, 'predicted-frame')
    #print(key)
    return StreamingHttpResponse(redis_camera(key), content_type='multipart/x-mixed-replace; boundary=frame')


@api_view(['GET'])
@renderer_classes((TemplateHTMLRenderer,JSONRenderer))
@csrf_exempt
def get_inspection_qc_list(request,process_id):
    from toyoda.utils import get_inspection_qc_list
    response = get_inspection_qc_list(process_id)
    return HttpResponse(json.dumps(response,cls=Encoder), content_type="application/json")