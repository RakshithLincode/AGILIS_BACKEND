from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from rest_framework.decorators import api_view, renderer_classes
from rest_framework.renderers import TemplateHTMLRenderer,JSONRenderer
from django.http import HttpResponse
from parts.utils import *
import json
from common.utils import *


from drf_yasg import openapi
from drf_yasg.openapi import Schema, TYPE_OBJECT, TYPE_STRING, TYPE_ARRAY
from drf_yasg.utils import swagger_auto_schema

@swagger_auto_schema(method='post', request_body=openapi.Schema(
    type=openapi.TYPE_OBJECT, 
    properties={
        'short_number' : openapi.Schema(type=openapi.TYPE_STRING, example='sht11'),
        'model_number' : openapi.Schema(type=openapi.TYPE_STRING, example='md11'),
        'part_number': openapi.Schema(type=openapi.TYPE_STRING, example='pt11'),
        'planned_production' : openapi.Schema(type=openapi.TYPE_STRING, example='100'),
        'part_description': openapi.Schema(type=openapi.TYPE_STRING, example='fjjff'),
        'edit_part_data' : openapi.Schema(type=openapi.TYPE_BOOLEAN, example='true')
    }
))

@api_view(['POST'])
@renderer_classes((TemplateHTMLRenderer,JSONRenderer))
@csrf_exempt
def add_part_details(request):
    data = json.loads(request.body)
    from parts.utils import add_part_details_task
    part_id = add_part_details_task(data)
    return HttpResponse(json.dumps({'message' : 'Part added Successfully!', 'part_id' : part_id}, cls=Encoder), content_type="application/json")

@api_view(['DELETE'])
@renderer_classes((TemplateHTMLRenderer,JSONRenderer))
@csrf_exempt
def delete_part(request, part_id):
    from parts.utils import delete_part_task
    delete_part_task(part_id)
    return HttpResponse(json.dumps({'message' : 'Part deleted Successfully!'}, cls=Encoder), content_type="application/json")


@swagger_auto_schema(method='patch', request_body=openapi.Schema(
    type=openapi.TYPE_OBJECT, 
    properties={
        '_id' : openapi.Schema(type=openapi.TYPE_STRING, example='5f32677047b362fbb536f1c0'),
        'short_number' : openapi.Schema(type=openapi.TYPE_STRING, example='sht11'),
        'model_number' : openapi.Schema(type=openapi.TYPE_STRING, example='md11'),
        'part_number': openapi.Schema(type=openapi.TYPE_STRING, example='pt11'),
        'planned_production' : openapi.Schema(type=openapi.TYPE_STRING, example='100'),
        'part_description': openapi.Schema(type=openapi.TYPE_STRING, example='fjjff'),
        'edit_part_data' : openapi.Schema(type=openapi.TYPE_BOOLEAN, example='true')
    }
))

@api_view(['PATCH'])
@renderer_classes((TemplateHTMLRenderer,JSONRenderer))
@csrf_exempt
def update_part(request):
    data = json.loads(request.body)
    from parts.utils import update_part_task
    response = update_part_task(data)
    return HttpResponse(json.dumps({'message' : 'Part updated Successfully!', 'part_id' : response}, cls=Encoder), content_type="application/json")

@api_view(['GET'])
@renderer_classes((TemplateHTMLRenderer,JSONRenderer))
@csrf_exempt
def get_part_details(request, part_id):
    from parts.utils import get_part_details_task
    response = get_part_details_task(part_id)
    return HttpResponse(json.dumps(response, cls=Encoder), content_type="application/json")


@api_view(['GET'])
@renderer_classes((TemplateHTMLRenderer,JSONRenderer))
@csrf_exempt
def get_parts(request):
    from parts.utils import get_parts_task 
    skip = request.GET.get('skip', 0)
    limit = request.GET.get('limit' , 10)
    response = get_parts_task(skip, limit)
    return HttpResponse(json.dumps(response, cls=Encoder), content_type="application/json")


@api_view(['GET'])
@renderer_classes((TemplateHTMLRenderer,JSONRenderer))
@csrf_exempt
def get_partInfo(request, short_number):
    from parts.utils import get_partInfo
    resp = get_partInfo(short_number)
    return HttpResponse(json.dumps(resp, cls=Encoder), content_type="application/json")


@api_view(['GET'])
@renderer_classes((TemplateHTMLRenderer,JSONRenderer))
@csrf_exempt
def get_short_numbers_list(request):
    from parts.utils import get_short_numbers_list_util 
    skip = request.GET.get('skip', 0)
    limit = request.GET.get('limit' , 10)
    response = get_short_numbers_list_util(skip, limit)
    return HttpResponse(json.dumps(response, cls=Encoder), content_type="application/json")
