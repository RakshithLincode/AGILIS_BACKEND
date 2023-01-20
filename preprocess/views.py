from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from rest_framework.decorators import api_view, renderer_classes
from rest_framework.renderers import TemplateHTMLRenderer
from common.utils import Encoder,RedisKeyBuilderServer,CacheHelper
import json
from django.http import HttpResponse,StreamingHttpResponse




@api_view(['POST'])
@renderer_classes((TemplateHTMLRenderer,))
@csrf_exempt
#@check_group(['admin'])
def set_crop(data):
    data = json.loads(data.body)
    from preprocess.utils import set_crop_util
    message,status_code = set_crop_util(data)
    if status_code == 200:
        return HttpResponse(json.dumps({'Message' : 'Success!', 'data' : message}, cls=Encoder), content_type="application/json")
    else:
        return HttpResponse( {message}, status=status_code)

@api_view(['GET'])
@csrf_exempt
def get_capture_feed_url(request):
    from preprocess.utils import get_camera_feed_urls
    url = get_camera_feed_urls()
    return HttpResponse(json.dumps( {'capture_url' : url} , cls=Encoder), content_type="application/json")


@api_view(['POST'])
@csrf_exempt
def initial_capture(request):
    data = json.loads(request.body)
    from preprocess.utils import initial_capture_util
    message,status_code = initial_capture_util(data)
    if status_code == 200:
        return HttpResponse(json.dumps({'Message' : 'Success!', 'data' : message}, cls=Encoder), content_type="application/json")
    else:
        return HttpResponse( {message}, status=status_code)


@api_view(['POST'])
@csrf_exempt
def set_init_regions(request):
    data = json.loads(request.body)
    from preprocess.utils import set_init_regions_util
    message,status_code = set_init_regions_util(data)
    if status_code == 200:
        return HttpResponse(json.dumps({'Message' : 'Success!', 'data' : message}, cls=Encoder), content_type="application/json")
    else:
        return HttpResponse( {message}, status=status_code)

@api_view(['POST'])
@csrf_exempt
def capture_util(request):
    data = json.loads(request.body)
    from preprocess.utils import capture_util
    message,status_code = capture_util(data)
    if status_code == 200:
        return HttpResponse(json.dumps({'Message' : 'Success!', 'data' : message}, cls=Encoder), content_type="application/json")
    else:
        return HttpResponse( {message}, status=status_code)

@api_view(['POST'])
@csrf_exempt
def final_capture(request):
    data = json.loads(request.body)
    from preprocess.utils import final_capture_util
    message,status_code = final_capture_util(data)
    if status_code == 200:
        return HttpResponse(json.dumps({'Message' : 'Success!', 'data' : message}, cls=Encoder), content_type="application/json")
    else:
        return HttpResponse( {message}, status=status_code)

@api_view(['GET'])
@csrf_exempt
def show_captured_img(request):
    data = json.loads(request.body)
    from preprocess.utils import show_captured_img_util
    message,status_code = show_captured_img_util(data)
    if status_code == 200:
        return HttpResponse(json.dumps({'Message' : 'Success!', 'data' : message}, cls=Encoder), content_type="application/json")
    else:
        return HttpResponse( {message}, status=status_code)

@api_view(['GET'])
@csrf_exempt
def change_img(request):
    data = json.loads(request.body)
    from preprocess.utils import change_img_util
    message,status_code = change_img_util(data)
    if status_code == 200:
        return HttpResponse(json.dumps({'Message' : 'Success!', 'data' : message}, cls=Encoder), content_type="application/json")
    else:
        return HttpResponse( {message}, status=status_code)

@api_view(['GET'])
@csrf_exempt
def get_camera_stream(request, wid, cameraid):
    from preprocess.utils import redis_camera
    key = RedisKeyBuilderServer(wid).get_key(cameraid, 'original-frame')
    print(key)
    return StreamingHttpResponse(redis_camera(key), content_type='multipart/x-mixed-replace; boundary=frame')


@api_view(['GET'])
@csrf_exempt
def get_redis_stream(request, key):
    from preprocess.utils import redis_camera
    #key = RedisKeyBuilderServer(wid).get_key(cameraid, 'predicted-frame')
    #print(key)
    return StreamingHttpResponse(redis_camera(key), content_type='multipart/x-mixed-replace; boundary=frame')

