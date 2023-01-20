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
##@check_group(['operator', 'admin'])
def get_dataset_list(request):
    from annotate.utils import get_dataset_list_util
    skip = request.GET.get('skip', 0)
    limit = request.GET.get('limit', 100) 
    message,status_code = get_dataset_list_util(skip, limit)
    #return HttpResponse(json.dumps(resp, cls=Encoder), content_type="application/json")
    #return HttpResponse(json.dumps({'Message' : 'Success!', 'data' : resp}, cls=Encoder), content_type="application/json")
    if status_code == 200:
        return HttpResponse(json.dumps([message], cls=Encoder))
    else:
        return HttpResponse( {message}, status=status_code)


@api_view(['GET'])
@csrf_exempt
#@check_group(['admin' , 'operator'])
def get_data_for_histogram(data):
    from annotate.utils import get_data_for_histogram_util
    message,status_code = get_data_for_histogram_util(data)
    if status_code == 200:
        return HttpResponse(json.dumps({'Message' : 'Success!', 'data' : message}, cls=Encoder), content_type="application/json")
    else:
        return HttpResponse( {message}, status=status_code)


@api_view(['POST'])
@renderer_classes((TemplateHTMLRenderer,))
@csrf_exempt
#@check_group(['admin'])
def create_dataset(data):
    from annotate.utils import create_dataset_util
    message,status_code = create_dataset_util(data)
    if status_code == 200:
        return HttpResponse(json.dumps({'Message' : 'Success!', 'data' : message}, cls=Encoder), content_type="application/json")
    else:
        return HttpResponse( {message}, status=status_code)

@api_view(['POST'])
@renderer_classes((TemplateHTMLRenderer,))
@csrf_exempt
#@check_group(['admin'])
def delete_img(data):
    data = json.loads(data.body)
    from annotate.utils import delete_img_util
    message,status_code = delete_img_util(data)
    if status_code == 200:
        return HttpResponse(json.dumps({'Message' : 'Success!', 'data' : message}, cls=Encoder), content_type="application/json")
    else:
        return HttpResponse( {message}, status=status_code)


@api_view(['POST'])
@csrf_exempt
#@check_group(['admin' , 'operator'])
def submit_annotations(request):
    cords_and_label = json.loads(request.body)
    from annotate.utils import submit_annotations_util
    message,status_code = submit_annotations_util(cords_and_label)
    if status_code == 200:
        return HttpResponse(json.dumps({'Message' : 'Success!', 'data' : message}, cls=Encoder), content_type="application/json")
    else:
        return HttpResponse( {message}, status=status_code)
    #return HttpResponse(json.dumps({'updated annotation details' : updated_anno}, cls=Encoder), content_type="application/json")

@api_view(['GET'])
@csrf_exempt
#@check_group(['admin' , 'operator'])
def fetch_data(data):
    from annotate.utils import fetch_data_util
    total,current,limit,message,status_code = fetch_data_util(data)
    if status_code == 200:
        return HttpResponse(json.dumps({'message' : 'Success!', 'data' : message,'total' : total,'current' : current,'limit' : limit}, cls=Encoder), content_type="application/json")
    else:
        return HttpResponse( {message}, status=status_code)
    #return HttpResponse(json.dumps({'list of annotations' : list_of_anno}, cls=Encoder), content_type="application/json")

@api_view(['POST'])
@csrf_exempt
#@check_group(['admin' , 'operator'])
def check_annotations(request):
    cords_and_label = json.loads(request.body)
    from annotate.utils import check_annotations_util
    message,status_code = check_annotations_util(cords_and_label)
    if status_code == 200:
        return HttpResponse(json.dumps({'Message' : 'Success!', 'data' : message}, cls=Encoder), content_type="application/json")
    else:
        return HttpResponse( {message}, status=status_code)
    #return HttpResponse(json.dumps({'updated annotation details' : updated_anno}, cls=Encoder), content_type="application/json")


@api_view(['GET'])                                                                                                                    
@csrf_exempt
#@check_group(['admin' , 'operator'])
def next_img(data):                                                                                                                                                       
    from annotate.utils import next_img_util
    message,status_code = next_img_util(data)
    if status_code == 200:
        return HttpResponse(json.dumps({'Message' : 'Success!', 'data' : message}, cls=Encoder), content_type="application/json")
    else:
        return HttpResponse( {message}, status=status_code)


@api_view(['GET'])                                                                                                                    
@csrf_exempt
#@check_group(['admin' , 'operator'])
def prev_img(data):                                                                                                                                                       
    from annotate.utils import prev_img_util
    message,status_code = prev_img_util(data)
    if status_code == 200:
        return HttpResponse(json.dumps({'Message' : 'Success!', 'data' : message}, cls=Encoder), content_type="application/json")
    else:
        return HttpResponse( {message}, status=status_code)


@api_view(['GET'])                                                                                                                    
@csrf_exempt
#@check_group(['admin' , 'operator'])
def get_img(data):                                                                                                                                                       
    from annotate.utils import get_img_util
    message,status_code = get_img_util(data)
    if status_code == 200:
        return HttpResponse(json.dumps({'Message' : 'Success!', 'data' : message}, cls=Encoder), content_type="application/json")
    else:
        return HttpResponse( {message}, status=status_code)


@api_view(['GET'])                                                                                                                    
@csrf_exempt
#@check_group(['admin' , 'operator'])
def fetch_image_url(data):                                                                                                                                                       
    from annotate.utils import fetch_image_url_util
    message,status_code = fetch_image_url_util(data)
    if status_code == 200:
        return HttpResponse(json.dumps({'Message' : 'Success!', 'data' : message}, cls=Encoder), content_type="application/json")
    else:
        return HttpResponse( {message}, status=status_code)
    #return HttpResponse(json.dumps({'list_of_urls' : list_of_urls}, cls=Encoder), content_type="application/json")


@api_view(['POST'])
@csrf_exempt
#@check_group(['admin' , 'operator'])
def export_data(request):
    cords_and_label = json.loads(request.body)
    from annotate.utils import export_data_utils
    message,status_code = export_data_utils(cords_and_label)
    if status_code == 200:
        return HttpResponse(json.dumps({'Message' : 'Success!', 'data' : message}, cls=Encoder), content_type="application/json")
    else:
        return HttpResponse( {message}, status=status_code)
    #return HttpResponse(json.dumps({'path of annotation' : path_of_annotation}, cls=Encoder), content_type="application/json")















##################################################


@api_view(['GET'])
@csrf_exempt
#@check_group(['admin' , 'operator'])
def card_flip_random_image(data):
    from annotate.utils import card_flip_random_image_util
    message,status_code = card_flip_random_image_util(data)
    if status_code == 200:
        return HttpResponse(json.dumps({'Message' : 'Success!', 'data' : message}, cls=Encoder), content_type="application/json")
    else:
        return HttpResponse( {message}, status=status_code)



@api_view(['POST'])
@csrf_exempt
#@check_group(['admin' , 'operator'])
def auto_salient_annotations(request):
    data = json.loads(request.body)
    from annotate.utils import auto_salient_annotations_utils
    pred_cords = auto_salient_annotations_utils(data)
    return HttpResponse(json.dumps({'predicted cords' : pred_cords}, cls=Encoder), content_type="application/json")


@api_view(['POST'])
@csrf_exempt
#@check_group(['admin' , 'operator'])
def auto_color_annotations(request):
    data = json.loads(request.body)
    from annotate.utils import auto_color_annotations_util
    pred_cords = auto_color_annotations_util(data)
    return HttpResponse(json.dumps({'predicted cords' : pred_cords}, cls=Encoder), content_type="application/json")


@api_view(['GET'])
@csrf_exempt
def get_capture_feed_url(request):
    from annotate.capture import get_capture_feed_url
    url = get_capture_feed_url()
    return HttpResponse(json.dumps( {'capture_url' : url} , cls=Encoder), content_type="application/json")


@api_view(['POST'])
@csrf_exempt
def capture_part_image(request):
    from annotate.capture import capture_part_image
    data = json.loads(request.body)
    part_id = data.get('part_id')
    if part_id:
        resp = capture_part_image(part_id)
        return HttpResponse(json.dumps( resp , cls=Encoder), content_type="application/json")
    return HttpResponse(json.dumps( {'message' : "Failed"} , cls=Encoder), content_type="application/json")
    



@api_view(['GET'])
@csrf_exempt
def start_camera(request):
    from annotate.capture import start_camera
    resp = start_camera()
    return HttpResponse(json.dumps( resp , cls=Encoder), content_type="application/json")


@api_view(['GET'])
@csrf_exempt
def stop_camera(request):
    from annotate.capture import stop_camera
    resp = stop_camera()
    return HttpResponse(json.dumps( resp , cls=Encoder), content_type="application/json")
