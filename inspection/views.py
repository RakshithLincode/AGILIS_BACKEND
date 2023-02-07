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
def get_current_inspection_details(data,jig_id):
    print(jig_id,'jig_id...............................')
    from inspection.tasks import get_current_inspection_details_utils
    message,status_code = get_current_inspection_details_utils(jig_id)
    if status_code == 200:
        return HttpResponse(json.dumps({'Message' : 'Success!', 'data' : message}, cls=Encoder), content_type="application/json")
    else:
        return HttpResponse( {message}, status=status_code)

@api_view(['GET'])
@renderer_classes((TemplateHTMLRenderer,))
@csrf_exempt
#@check_group(['admin'])
def get_running_process(data):
    from inspection.tasks import get_running_process
    message,status_code = get_running_process()
    # print(message,'hhhhhhhddddddddddhhhhhhhhhhhhhhhhhhhh')
    if status_code == 200:
        return HttpResponse(json.dumps({'Message' : 'Success!', 'data' : message}, cls=Encoder), content_type="application/json")
    else:
        return HttpResponse( {message}, status=status_code)


@api_view(['POST'])
@csrf_exempt
def start_process_schneider(request):
    config = json.loads(request.body)
    from inspection.tasks import start_inspection, start_real_inspection
    resp,inspection_id = start_inspection(config)
    if inspection_id is None:
        return HttpResponse(json.dumps({'Message' : 'Failed!', 'data' : resp}, cls=Encoder), content_type="application/json")
    inspection_id = str(inspection_id)
    print(config,'fggggggggggvfhvhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhh')
    start_real_inspection.delay(config, inspection_id)
    return HttpResponse(json.dumps({'Message' : 'Success!', 'data' : resp}, cls=Encoder), content_type="application/json")


@api_view(['POST'])
@csrf_exempt
def force_admin_pass(request):

    config = json.loads(request.body)
    from inspection.tasks import force_admin_pass
    message,status_code = force_admin_pass(config)
    if status_code == 200:
        return HttpResponse(json.dumps({'Message' : 'Success!', 'data' : message}, cls=Encoder), content_type="application/json")
    else:
        return HttpResponse( {message}, status=status_code)

@api_view(['POST'])
@csrf_exempt
def reject_part(request):

    config = json.loads(request.body)
    from inspection.tasks import reject_part
    message,status_code = reject_part(config)
    if status_code == 200:
        return HttpResponse(json.dumps({'Message' : 'Success!', 'data' : message}, cls=Encoder), content_type="application/json")
    else:
        return HttpResponse( {message}, status=status_code)


@api_view(['POST'])
@csrf_exempt
def continue_process(request):

    config = json.loads(request.body)
    from inspection.tasks import continue_process
    message,status_code = continue_process(config)
    if status_code == 200:
        return HttpResponse(json.dumps({'Message' : 'Success!', 'data' : message}, cls=Encoder), content_type="application/json")
    else:
        return HttpResponse( {message}, status=status_code)


@api_view(['GET'])
@renderer_classes((TemplateHTMLRenderer,))
@csrf_exempt
#@check_group(['admin'])
def get_static(data,jig_id):
    from inspection.tasks import get_static
    message,status_code = get_static(jig_id)
    if status_code == 200:
        return HttpResponse(json.dumps({'Message' : 'Success!', 'data' : message}, cls=Encoder), content_type="application/json")
    else:
        return HttpResponse( {message}, status=status_code)

@api_view(['GET'])
@renderer_classes((TemplateHTMLRenderer,))
@csrf_exempt
#@check_group(['admin'])
def get_process_retry(data,jig_id):
    from inspection.tasks import get_process_retry
    message,status_code = get_process_retry(jig_id)
    if status_code == 200:
        return HttpResponse(json.dumps({'Message' : 'Success!', 'data' : message}, cls=Encoder), content_type="application/json")
    else:
        return HttpResponse( {message}, status=status_code)
        
@api_view(['POST'])
@renderer_classes((TemplateHTMLRenderer,))
@csrf_exempt
#@check_group(['admin'])
def admin_report_reset(data):
    
    data = json.loads(data.body)
    from inspection.tasks import admin_report_reset
    message,status_code = admin_report_reset(data)
    if status_code == 200:
        return HttpResponse(json.dumps({'Message' : 'Success!', 'data' : message}, cls=Encoder), content_type="application/json")
    else:
        return HttpResponse( {message}, status=status_code)
