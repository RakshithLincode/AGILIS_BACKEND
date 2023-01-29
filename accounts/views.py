from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from rest_framework.decorators import api_view, renderer_classes
from rest_framework.renderers import TemplateHTMLRenderer,JSONRenderer
from django.http import HttpResponse
import json
from common.utils import Encoder
from rest_framework import status
from rest_framework.response import Response

from drf_yasg import openapi
from drf_yasg.openapi import Schema, TYPE_OBJECT, TYPE_STRING, TYPE_ARRAY
from drf_yasg.utils import swagger_auto_schema

################################################################USER CRUDS################################################################

@swagger_auto_schema(method='post', request_body=openapi.Schema(
    type=openapi.TYPE_OBJECT, 
    properties={

        'username' : openapi.Schema(type=openapi.TYPE_STRING, example='xyz'),
        'first_name' : openapi.Schema(type=openapi.TYPE_STRING, example='x'),
        'last_name': openapi.Schema(type=openapi.TYPE_STRING, example='yz'),
        'email' : openapi.Schema(type=openapi.TYPE_STRING, example='xyz@xyz.com'),
        'is_staff': openapi.Schema(type=openapi.TYPE_BOOLEAN, example=True),
        'user_address' : openapi.Schema(type=openapi.TYPE_STRING, example='abc'),
        'is_superuser': openapi.Schema(type=openapi.TYPE_BOOLEAN, example=False),
        'role_name' : openapi.Schema(type=openapi.TYPE_STRING, example='supervisor'),
        'phone_number' : openapi.Schema(type=openapi.TYPE_STRING, example='43283743113'),
        'password' : openapi.Schema(type=openapi.TYPE_STRING, example='password')

    }
))
@api_view(['POST'])
@renderer_classes((TemplateHTMLRenderer,JSONRenderer))
@csrf_exempt
def add_user_account(request):
    data = json.loads(request.body)
    from accounts.utils import add_user_account_util
    message = add_user_account_util(data)
    return HttpResponse(json.dumps({'message' : message}), content_type="application/json") 


@api_view(['GET'])
@renderer_classes((TemplateHTMLRenderer,JSONRenderer))
@csrf_exempt
def get_user_account(request,client_id):
    from accounts.utils import get_user_account_util
    response = get_user_account_util(client_id)
    return HttpResponse(json.dumps(response), content_type="application/json") 


@swagger_auto_schema(method='patch', request_body=openapi.Schema(
    type=openapi.TYPE_OBJECT, 
    properties={

        'user_id' : openapi.Schema(type=openapi.TYPE_STRING, example='d92719dd-cd03-4857-bf7f-7b78a68ba888'),
        'first_name' : openapi.Schema(type=openapi.TYPE_STRING, example='x'),
        'last_name': openapi.Schema(type=openapi.TYPE_STRING, example='yz'),
        'email' : openapi.Schema(type=openapi.TYPE_STRING, example='xyz@xyz.com'),
        'is_staff': openapi.Schema(type=openapi.TYPE_BOOLEAN, example=True),
        'user_address' : openapi.Schema(type=openapi.TYPE_STRING, example='abc'),
        'is_superuser': openapi.Schema(type=openapi.TYPE_BOOLEAN, example=False),
        'role_name' : openapi.Schema(type=openapi.TYPE_STRING, example='operator'),
        'phone_number' : openapi.Schema(type=openapi.TYPE_STRING, example='43283743113'),
        'password' : openapi.Schema(type=openapi.TYPE_STRING, example='password')

    }
))
@api_view(['PATCH'])
@renderer_classes((TemplateHTMLRenderer,JSONRenderer))
@csrf_exempt
def update_user_account(request):
    data = json.loads(request.body)
    from accounts.utils import update_user_account_util
    message = update_user_account_util(data)
    return HttpResponse(json.dumps({'message' : message}), content_type="application/json")


@api_view(['DELETE'])
@renderer_classes((TemplateHTMLRenderer,JSONRenderer))
@csrf_exempt
def delete_user_account(request, client_id):
    from accounts.utils import delete_user_account_util
    message = delete_user_account_util(client_id)
    return HttpResponse(json.dumps({'message' : message}), content_type="application/json")


@api_view(['GET'])
@renderer_classes((TemplateHTMLRenderer,JSONRenderer))
@csrf_exempt
def get_all_user_accounts(request):
    from accounts.utils import get_all_user_accounts_util
    response = get_all_user_accounts_util()
    return  HttpResponse(json.dumps(response), content_type="application/json")


#########################################################USER_SI CRUDS############################################################


@swagger_auto_schema(method='post', request_body=openapi.Schema(
    type=openapi.TYPE_OBJECT, 
    properties={
        'si_id' : openapi.Schema(type=openapi.TYPE_STRING, example='e396a732-c187-477a-8d94-7a4def2f1678'),
        'username' : openapi.Schema(type=openapi.TYPE_STRING, example='xyz'),
        'first_name' : openapi.Schema(type=openapi.TYPE_STRING, example='x'),
        'last_name': openapi.Schema(type=openapi.TYPE_STRING, example='yz'),
        'email' : openapi.Schema(type=openapi.TYPE_STRING, example='xyz@xyz.com'),
        'is_staff': openapi.Schema(type=openapi.TYPE_BOOLEAN, example=True),
        'user_address' : openapi.Schema(type=openapi.TYPE_STRING, example='abc'),
        'is_superuser': openapi.Schema(type=openapi.TYPE_BOOLEAN, example=False),
        'role_name' : openapi.Schema(type=openapi.TYPE_STRING, example='34734'),
        'phone_number' : openapi.Schema(type=openapi.TYPE_STRING, example='43283743113'),
        'password' : openapi.Schema(type=openapi.TYPE_STRING, example='password')

    }
))
@api_view(['POST'])
@renderer_classes((TemplateHTMLRenderer,JSONRenderer))
@csrf_exempt
def add_user_si(request):
    data = json.loads(request.body)
    from accounts.utils import add_user_si_util
    message = add_user_si_util(data)
    return HttpResponse(json.dumps({'message' : message}), content_type="application/json") 


@swagger_auto_schema(method='patch', request_body=openapi.Schema(
    type=openapi.TYPE_OBJECT, 
    properties={
        'si_id' : openapi.Schema(type=openapi.TYPE_STRING, example='e396a732-c187-477a-8d94-7a4def2f1678'),
        'username' : openapi.Schema(type=openapi.TYPE_STRING, example='xyz'),
        'first_name' : openapi.Schema(type=openapi.TYPE_STRING, example='x'),
        'last_name': openapi.Schema(type=openapi.TYPE_STRING, example='yz'),
        'email' : openapi.Schema(type=openapi.TYPE_STRING, example='xyz@xyz.com'),
        'is_staff': openapi.Schema(type=openapi.TYPE_BOOLEAN, example=True),
        'user_address' : openapi.Schema(type=openapi.TYPE_STRING, example='abc'),
        'is_superuser': openapi.Schema(type=openapi.TYPE_BOOLEAN, example=False),
        'role_name' : openapi.Schema(type=openapi.TYPE_STRING, example='34734'),
        'phone_number' : openapi.Schema(type=openapi.TYPE_STRING, example='43283743113'),
        'password' : openapi.Schema(type=openapi.TYPE_STRING, example='password')

    }
))
@api_view(['PATCH'])
@renderer_classes((TemplateHTMLRenderer,JSONRenderer))
@csrf_exempt
def update_user_si(request):
    data = json.loads(request.body)
    from accounts.utils import update_user_si_util
    message = update_user_si_util(data)
    return HttpResponse(json.dumps({'message' : message}), content_type="application/json")


@api_view(['DELETE'])
@renderer_classes((TemplateHTMLRenderer,JSONRenderer))
@csrf_exempt
def delete_user_si(request, client_id):
    from accounts.utils import delete_user_si_util
    message = delete_user_si_util(client_id)
    return HttpResponse(json.dumps({'message' : message}), content_type="application/json")

@api_view(['GET'])
@renderer_classes((TemplateHTMLRenderer,JSONRenderer))
@csrf_exempt
def get_user_si(request,client_id):
    from accounts.utils import get_user_si_util
    response = get_user_si_util(client_id)
    return HttpResponse(json.dumps(response), content_type="application/json") 


@api_view(['GET'])
@renderer_classes((TemplateHTMLRenderer,JSONRenderer))
@csrf_exempt
def get_user_sis(request):
    from accounts.utils import get_user_sis_util
    response = get_user_sis_util()
    return  HttpResponse(json.dumps(response), content_type="application/json") 


################################################################USER_CLIENT CRUDS#############################################################

@swagger_auto_schema(method='post', request_body=openapi.Schema(
    type=openapi.TYPE_OBJECT, 
    properties={
        'client_id' : openapi.Schema(type=openapi.TYPE_STRING, example='e396a732-c187-477a-8d94-7a4def2f1678'),
        'username' : openapi.Schema(type=openapi.TYPE_STRING, example='xyz'),
        'first_name' : openapi.Schema(type=openapi.TYPE_STRING, example='x'),
        'last_name': openapi.Schema(type=openapi.TYPE_STRING, example='yz'),
        'email' : openapi.Schema(type=openapi.TYPE_STRING, example='xyz@xyz.com'),
        'is_staff': openapi.Schema(type=openapi.TYPE_BOOLEAN, example=True),
        'user_address' : openapi.Schema(type=openapi.TYPE_STRING, example='abc'),
        'is_superuser': openapi.Schema(type=openapi.TYPE_BOOLEAN, example=False),
        'role_name' : openapi.Schema(type=openapi.TYPE_STRING, example='34734'),
        'phone_number' : openapi.Schema(type=openapi.TYPE_STRING, example='43283743113'),
        'client_user_manager' : openapi.Schema(type=openapi.TYPE_STRING, example='None'),
        'password' : openapi.Schema(type=openapi.TYPE_STRING, example='password'),
        'is_client_admin' : openapi.Schema(type=openapi.TYPE_BOOLEAN, example=True)

    }
))
@api_view(['POST'])
@renderer_classes((TemplateHTMLRenderer,JSONRenderer))
@csrf_exempt
def add_user_client(request):
    # data = json.loads(request.body)
    data = json.loads(request.body)
    from accounts.utils import add_user_client_util
    message = add_user_client_util(data)
    return HttpResponse(json.dumps({'message' : message}), content_type="application/json") 


@swagger_auto_schema(method='patch', request_body=openapi.Schema(
    type=openapi.TYPE_OBJECT, 
    properties={
        'user_id' : openapi.Schema(type=openapi.TYPE_STRING, example='be5f1c5d-b5aa-457a-8cf9-9112307e9c38'),
        'first_name' : openapi.Schema(type=openapi.TYPE_STRING, example='x'),
        'last_name': openapi.Schema(type=openapi.TYPE_STRING, example='yz'),
        'email' : openapi.Schema(type=openapi.TYPE_STRING, example='xyz@xyz.com'),
        'is_staff': openapi.Schema(type=openapi.TYPE_BOOLEAN, example=True),
        'user_address' : openapi.Schema(type=openapi.TYPE_STRING, example='abc'),
        'is_superuser': openapi.Schema(type=openapi.TYPE_BOOLEAN, example=False),
        'role_name' : openapi.Schema(type=openapi.TYPE_STRING, example='supervisor'),
        'phone_number' : openapi.Schema(type=openapi.TYPE_STRING, example='43283743113'),
        'client_user_manager' : openapi.Schema(type=openapi.TYPE_STRING, example='None'),
        'password' : openapi.Schema(type=openapi.TYPE_STRING, example='password'),
        'is_client_admin' : openapi.Schema(type=openapi.TYPE_BOOLEAN, example=True)

    }
))
@api_view(['PATCH'])
@renderer_classes((TemplateHTMLRenderer,JSONRenderer))
@csrf_exempt
def update_user_client(request):
    data = json.loads(request.body)
    from accounts.utils import update_user_client_util
    message = update_user_client_util(data)
    return HttpResponse(json.dumps({'message' : message}), content_type="application/json")


@api_view(['DELETE'])
@renderer_classes((TemplateHTMLRenderer,JSONRenderer))
@csrf_exempt
def delete_user_client(request, client_id):
    from accounts.utils import delete_user_client_util
    message = delete_user_client_util(client_id)
    return HttpResponse(json.dumps({'message' : message}), content_type="application/json")


@api_view(['GET'])
@renderer_classes((TemplateHTMLRenderer,JSONRenderer))
@csrf_exempt
def get_user_client(request,client_id):
    from accounts.utils import get_user_client_util
    response = get_user_client_util(client_id)
    return HttpResponse(json.dumps(response), content_type="application/json") 


@api_view(['GET'])
@renderer_classes((TemplateHTMLRenderer,JSONRenderer))
@csrf_exempt
def get_user_clients(request):
    from accounts.utils import get_user_clients_util
    response = get_user_clients_util()
    return  HttpResponse(json.dumps(response), content_type="application/json") 


################################################################CLIENT CRUDS################################################################


@swagger_auto_schema(method='post', request_body=openapi.Schema(
    type=openapi.TYPE_OBJECT, 
    properties={
        'client_name' : openapi.Schema(type=openapi.TYPE_STRING, example='Toyoda'),
        'client_license_key' : openapi.Schema(type=openapi.TYPE_STRING, example='dfhiweuf8w'),
        'client_role' : openapi.Schema(type=openapi.TYPE_STRING, example='Role2'),
        'assigned_to': openapi.Schema(type=openapi.TYPE_STRING, example='Client_User3'),
        'client_address' : openapi.Schema(type=openapi.TYPE_STRING, example='Bangalore')
    }
))
@api_view(['POST'])
@renderer_classes((TemplateHTMLRenderer,JSONRenderer))
@csrf_exempt
def add_client_account(request):
    print(request)
    data = json.loads(request.body)
    from accounts.utils import add_client_account_util
    message = add_client_account_util(data)
    return HttpResponse(json.dumps({'message' : message}), content_type="application/json") 


@api_view(['GET'])
@renderer_classes((TemplateHTMLRenderer,JSONRenderer))
@csrf_exempt
def get_client_account(request,client_id):
    from accounts.utils import get_client_account_util
    response = get_client_account_util(client_id)
    return HttpResponse(json.dumps(response), content_type="application/json") 


@swagger_auto_schema(method='patch', request_body=openapi.Schema(
    type=openapi.TYPE_OBJECT, 
    properties={
        'client_id' : openapi.Schema(type=openapi.TYPE_STRING, example='96e993fc-3f24-4e06-b2d1-e8736882c27f'),
        'client_name' : openapi.Schema(type=openapi.TYPE_STRING, example='Toyoda'),
        'client_license_key' : openapi.Schema(type=openapi.TYPE_STRING, example='dfhiweuf8w'),
        'client_role' : openapi.Schema(type=openapi.TYPE_STRING, example='Role2'),
        'assigned_to': openapi.Schema(type=openapi.TYPE_STRING, example='Client_User3'),
        'client_address' : openapi.Schema(type=openapi.TYPE_STRING, example='Bangalore')
    }
))
@api_view(['PATCH'])
@renderer_classes((TemplateHTMLRenderer,JSONRenderer))
@csrf_exempt
def update_client_account(request):
    data = json.loads(request.body)
    from accounts.utils import update_client_account_util
    message = update_client_account_util(data)
    return HttpResponse(json.dumps({'message' : message}), content_type="application/json")


@api_view(['DELETE'])
@renderer_classes((TemplateHTMLRenderer,JSONRenderer))
@csrf_exempt
def delete_client_account(request, client_id):
    from accounts.utils import delete_client_account_util
    message = delete_client_account_util(client_id)
    return HttpResponse(json.dumps({'message' : message}), content_type="application/json")


@api_view(['GET'])
@renderer_classes((TemplateHTMLRenderer,JSONRenderer))
@csrf_exempt
def get_all_client_accounts(request):
    from accounts.utils import get_all_client_accounts_util
    response = get_all_client_accounts_util()
    return  HttpResponse(json.dumps(response), content_type="application/json") 


################################################################SI CRUDS################################################################



@swagger_auto_schema(method='post', request_body=openapi.Schema(
    type=openapi.TYPE_OBJECT, 
    properties={
        'client_name' : openapi.Schema(type=openapi.TYPE_STRING, example='Toyoda'),
        'client_license_key' : openapi.Schema(type=openapi.TYPE_STRING, example='dfhiweuf8w'),
        'client_role' : openapi.Schema(type=openapi.TYPE_STRING, example='Role2'),
        'assigned_to': openapi.Schema(type=openapi.TYPE_STRING, example='Client_User3'),
        'client_address' : openapi.Schema(type=openapi.TYPE_STRING, example='Bangalore'),
        'activeClients' : openapi.Schema(type=openapi.TYPE_STRING, example='Cataler'),
        'license_key_list' : openapi.Schema(type=openapi.TYPE_STRING, example='qewerade')
    }
))
@api_view(['POST'])
@renderer_classes((TemplateHTMLRenderer,JSONRenderer))
@csrf_exempt
def add_si_account(request):
    data = json.loads(request.body)
    from accounts.utils import add_si_account_util
    message = add_si_account_util(data)
    return HttpResponse(json.dumps({'message' : message}), content_type="application/json") 


@api_view(['GET'])
@renderer_classes((TemplateHTMLRenderer,JSONRenderer))
@csrf_exempt
def get_si_account(request,client_id):
    from accounts.utils import get_si_account_util
    response = get_si_account_util(client_id)
    return HttpResponse(json.dumps(response), content_type="application/json") 


@swagger_auto_schema(method='patch', request_body=openapi.Schema(
    type=openapi.TYPE_OBJECT, 
    properties={
        'client_id' : openapi.Schema(type=openapi.TYPE_STRING, example='Toyoda'),
        'client_name' : openapi.Schema(type=openapi.TYPE_STRING, example='16523a25-897b-4dc3-94e8-48b4d94e1be9'),
        'client_license_key' : openapi.Schema(type=openapi.TYPE_STRING, example='dfhiweuf8w'),
        'client_role' : openapi.Schema(type=openapi.TYPE_STRING, example='Role2'),
        'assigned_to': openapi.Schema(type=openapi.TYPE_STRING, example='Client_User3'),
        'client_address' : openapi.Schema(type=openapi.TYPE_STRING, example='Bangalore'),
        'activeClients' : openapi.Schema(type=openapi.TYPE_STRING, example='Cataler'),
        'license_key_list' : openapi.Schema(type=openapi.TYPE_STRING, example='qewerade')
    }
))
@api_view(['PATCH'])
@renderer_classes((TemplateHTMLRenderer,JSONRenderer))
@csrf_exempt
def update_si_account(request):
    data = json.loads(request.body)
    from accounts.utils import update_si_account_util
    message = update_si_account_util(data)
    return HttpResponse(json.dumps({'message' : message}), content_type="application/json")


@api_view(['DELETE'])
@renderer_classes((TemplateHTMLRenderer,JSONRenderer))
@csrf_exempt
def delete_si_account(request, client_id):
    from accounts.utils import delete_si_account_util
    message = delete_si_account_util(client_id)
    return HttpResponse(json.dumps({'message' : message}), content_type="application/json")


@api_view(['GET'])
@renderer_classes((TemplateHTMLRenderer,JSONRenderer))
@csrf_exempt
def get_all_si_accounts(request):
    from accounts.utils import get_all_si_accounts_util
    response = get_all_si_accounts_util()
    return  HttpResponse(json.dumps(response), content_type="application/json") 


##############################################User Authentication APIS##########################################

@swagger_auto_schema(method='post', request_body=openapi.Schema(
    type=openapi.TYPE_OBJECT, 
    properties={
        'email' : openapi.Schema(type=openapi.TYPE_STRING, example='sysadmin_xyz@xyz.com'),
        'password' : openapi.Schema(type=openapi.TYPE_STRING, example='password'),
        'workstation_name' : openapi.Schema(type=openapi.TYPE_STRING, example='W_01')
    }
))
@api_view(['POST'])
@renderer_classes((TemplateHTMLRenderer,JSONRenderer))
@csrf_exempt
def login_user(request):
    data = json.loads(request.body)
    from accounts.utils import login_user_util
    message = login_user_util(data)
    if message == "MissingField":
        return HttpResponse( {'Missing fields'}, status=status.HTTP_401_UNAUTHORIZED)
    elif message == "AuthError":
        return HttpResponse({'Authentication failed'}, status=status.HTTP_401_UNAUTHORIZED)
    return  HttpResponse(json.dumps(message,cls=Encoder), content_type="application/json")


@swagger_auto_schema(method='post', request_body=openapi.Schema(
    type=openapi.TYPE_OBJECT, 
    properties={
        'email' : openapi.Schema(type=openapi.TYPE_STRING, example='sysadmin_xyz@xyz.com'),
        'password' : openapi.Schema(type=openapi.TYPE_STRING, example='password'),
        'workstation_name' : openapi.Schema(type=openapi.TYPE_STRING, example='W_01')
    }
))
@api_view(['POST'])
@renderer_classes((TemplateHTMLRenderer,JSONRenderer))
@csrf_exempt
def login_supervisor(request):
    data = json.loads(request.body)
    from accounts.utils import login_supervisor_util
    message = login_supervisor_util(data)
    if message == "MissingField":
        return HttpResponse( {'Missing fields'}, status=status.HTTP_401_UNAUTHORIZED)
    elif message == "AuthError":
        return HttpResponse({'Authentication failed'}, status=status.HTTP_401_UNAUTHORIZED)
    return  HttpResponse(json.dumps({'message' : message}), content_type="application/json") 
    
@api_view(['POST'])
@renderer_classes((TemplateHTMLRenderer,JSONRenderer))
@csrf_exempt
def change_password(request):
    data = json.loads(request.body)
    
    from accounts.utils import change_password_util
    message,message1 = change_password_util(data,request)
    if message1 == "fail":
        return HttpResponse({message}, status=500)
    else:
        return  HttpResponse(json.dumps(message,cls=Encoder), content_type="application/json")

