from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from rest_framework.decorators import api_view, renderer_classes
from rest_framework.renderers import TemplateHTMLRenderer
from django.http import HttpResponse
from common.utils import Encoder
import json


##########

@api_view(['GET'])
@renderer_classes((TemplateHTMLRenderer,))
@csrf_exempt
#@check_group(['admin'])
def list_specific_jig(request,jig_type):
    from configuration.utils import list_specific_jig
    message,status_code = list_specific_jig(jig_type)
    print(message,'messsssssssssssssssssssssssssssssssssssssssssssssssssssss')
    if status_code == 200:
        return HttpResponse(json.dumps({'Message' : 'Success!', 'data' : message}, cls=Encoder), content_type="application/json")
    else:
        return HttpResponse( {message}, status=status_code)

@api_view(['GET'])
@renderer_classes((TemplateHTMLRenderer,))
@csrf_exempt
#@check_group(['admin'])
def fetch_individual_component_list(data):
    from configuration.utils import fetch_individual_component_list_util
    message,status_code = fetch_individual_component_list_util(data)
    if status_code == 200:
        return HttpResponse(json.dumps({'Message' : 'Success!', 'data' : message}, cls=Encoder), content_type="application/json")
    else:
        return HttpResponse( {message}, status=status_code)


@api_view(['GET'])
@renderer_classes((TemplateHTMLRenderer,))
@csrf_exempt
#@check_group(['admin'])
def fetch_jig_list(data):
    from configuration.utils import fetch_jig_list_util
    message,status_code = fetch_jig_list_util(data)
    if status_code == 200:
        return HttpResponse(json.dumps({'Message' : 'Success!', 'data' : message}, cls=Encoder), content_type="application/json")
    else:
        return HttpResponse( {message}, status=status_code)

@api_view(['POST'])
@renderer_classes((TemplateHTMLRenderer,))
@csrf_exempt
#@check_group(['admin'])
def add_jig(data):
    data = json.loads(data.body)
    from configuration.utils import add_jig_util
    message,status_code = add_jig_util(data)
    if status_code == 200:
        return HttpResponse(json.dumps({'Message' : 'Success!', 'data' : message}, cls=Encoder), content_type="application/json")
    else:
        return HttpResponse( {message}, status=status_code)


@api_view(['GET'])
@renderer_classes((TemplateHTMLRenderer,))
@csrf_exempt
#@check_group(['admin'])
def fetch_specific_jig(request,jig_id):
    from configuration.utils import fetch_specific_jig_util
    message,status_code = fetch_specific_jig_util(jig_id)
    if status_code == 200:
        return HttpResponse(json.dumps({'Message' : 'Success!', 'data' : message}, cls=Encoder), content_type="application/json")
    else:
        return HttpResponse( {message}, status=status_code)

@api_view(['POST'])
@renderer_classes((TemplateHTMLRenderer,))
@csrf_exempt
#@check_group(['admin'])
def update_jig(data):
    data = json.loads(data.body)
    from configuration.utils import update_jig_util
    message,status_code = update_jig_util(data)
    if status_code == 200:
        return HttpResponse(json.dumps({'Message' : 'Success!', 'data' : message}, cls=Encoder), content_type="application/json")
    else:
        return HttpResponse( {message}, status=status_code)

@api_view(['POST'])
@renderer_classes((TemplateHTMLRenderer,))
@csrf_exempt
#@check_group(['admin'])
def delete_jig(data):
    data = json.loads(data.body)
    from configuration.utils import delete_jig_util
    message,status_code = delete_jig_util(data)
    if status_code == 200:
        return HttpResponse(json.dumps({'Message' : 'Success!', 'data' : message}, cls=Encoder), content_type="application/json")
    else:
        return HttpResponse( {message}, status=status_code)
