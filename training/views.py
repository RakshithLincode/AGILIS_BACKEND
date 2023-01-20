import json
from django.views.decorators.csrf import csrf_exempt
from django.http import HttpResponse
from rest_framework.decorators import api_view
from common.utils import Encoder
from pathlib import Path
from common.utils import MongoHelper
from training.tasks import process_job_request

from drf_yasg import openapi
from drf_yasg.openapi import Schema, TYPE_OBJECT, TYPE_STRING, TYPE_ARRAY
from drf_yasg.utils import swagger_auto_schema


@api_view(['GET'])
@csrf_exempt
def get_trained_list(request):
    from training.tasks import get_trained_list_util
    response = get_trained_list_util()
    return HttpResponse(json.dumps(response, cls=Encoder), content_type="application/json")


@swagger_auto_schema(method='post', request_body=openapi.Schema(
    type=openapi.TYPE_OBJECT, 
    properties={
        'experiment_name' : openapi.Schema(type=openapi.TYPE_STRING, example='test_tf_1'),
        'part_id': openapi.Schema(type=openapi.TYPE_STRING, example='5f42b5a90b72e674c03691d5'),
        'selected_labels' : openapi.Schema(type=openapi.TYPE_STRING, example='["Classification", "Hole", "hole1"]'),
        'type' : openapi.Schema(type=openapi.TYPE_STRING, example='tf'),
    }
))



@api_view(['POST'])
@csrf_exempt
def create_experiment(request):
    config = json.loads(request.body)
    from training.tasks import process_job_request, add_experiment
    experiment_id,part_id = add_experiment(config)
    experiment_id = str(experiment_id)
    part_id = str(part_id)
    process_job_request.delay(config, experiment_id,part_id)
    response = {'experiment_id': experiment_id}
    return HttpResponse(json.dumps(response, cls=Encoder), content_type="application/json")

@api_view(['GET'])
@csrf_exempt
def get_experiment_status(request,part_id, experiment_id):
    from training.tasks import get_experiment_status
    response = get_experiment_status(part_id, experiment_id)
    return HttpResponse(json.dumps(response, cls=Encoder), content_type="application/json")

@api_view(['GET'])
@csrf_exempt
def get_running_experiment(request,part_id):
    from training.tasks import get_running_experiment_status
    response = get_running_experiment_status(part_id)
    return HttpResponse(json.dumps(response, cls=Encoder), content_type="application/json")

@api_view(['GET'])
@csrf_exempt
def get_all_running_experiments(request):
    from training.tasks import get_all_running_experiments_status
    response = get_all_running_experiments_status()
    return HttpResponse(json.dumps(response, cls=Encoder), content_type="application/json")

@api_view(['POST'])
@csrf_exempt
def deploy_experiment(request):
    deployment_task = json.loads(request.body)
    from training.tasks import deploy_experiment_util
    deployment_status = deploy_experiment_util(deployment_task)
    return HttpResponse(json.dumps({'status' : 'Success!', 'deployment_status' : deployment_status}, cls=Encoder), content_type="application/json")


@api_view(['GET'])
@csrf_exempt
def get_deployment_list(request):
    from training.tasks import get_deployment_list_util
    response = get_deployment_list_util()
    return HttpResponse(json.dumps(response, cls=Encoder), content_type="application/json")


@api_view(['GET'])
@csrf_exempt
def monk_list_models(request):
    from training.tasks import monk_list_models_utils
    message,status_code = monk_list_models_utils(request)
    return HttpResponse(json.dumps({'status' : 'Success!', 'message' : message}, cls=Encoder), content_type="application/json")


"""
@api_view(['POST'])
@csrf_exempt
def run_monk_train(request):
    params = json.loads(request.body)
    from training.tasks import run_monk_train_utils
    message,status_code = run_monk_train_utils(params)
    return HttpResponse(json.dumps({'status' : 'Success!', 'message' : message}, cls=Encoder), content_type="application/json")
"""
