from accounts.views import *
from accounts.models import User,User_Client,User_SI,Client,SI
from django.core import serializers
from django.forms.models import model_to_dict
import uuid
from django.utils import timezone
from datetime import datetime
from django.contrib.auth import authenticate
from common.utils import *
from livis.settings import *
from livis.constants import *
from datetime import datetime


################################################################USER CRUDS################################################################

def change_password_util(data,request):

    user_ip_old_password = data.get('old_password',None)
    new_password = data.get('new_password',None)
    reenter_new_password = data.get('reenter_new_password',None)
    
    username = 'Admin'
    
    from django.contrib.auth import get_user_model
    User = get_user_model()
    u = User.objects.get(username=username)
    

    if u.check_password(user_ip_old_password) is False:
        message = "Invalid Old Password"
        message1 = 'fail'
        return message,message1
        
    if str(new_password) == str(user_ip_old_password):
        message = "New password cannot be same as old password"
        message1 = 'fail'
        return message,message1
    if str(new_password) != str(reenter_new_password):
        message = "password doesn't match"
        message1 = 'fail'
        return message,message1   
           

    u.set_password(new_password)
    u.save()
    
    message = "Password Changed, Please logout and relogin"
    message1 = 'pass'
    return message,message1

def add_user_account_util(data):
    """
    Usage: API to add a user account.
    Request Parameters: {
	    "username" : "shivu232",
	    "first_name" : "Shivani",
	    "last_name" : "Udupa",
	    "email" : "shivani@gmail.com",
	    "is_staff" : false,
	    "user_address" : "Bangalore",
        "is_superuser" : false,
        "role_name" : "122did",
        "phone_number" : 21218739283,
        "password" : "password"
    }
    Request Method: POST
    Response: {
        "message": "User Account added successfully."
    }
    """
    resp = {}
    try:
        user_id = str(uuid.uuid4()) 
        username = data.get('username',None)
        first_name = data.get('first_name',None)
        # last_name = data.get('last_name',None)
        is_staff = data.get('is_staff',None)
        email = data.get('email',None)
        # user_address = data.get('user_address',None)
        date_joined = timezone.now()
        updated_at = timezone.now()
        is_deleted = False
        is_active = True
        is_superuser = data.get('is_superuser',None)
        role_name = data.get('role_name',None)
        # phone_number = data.get('phone_number',None)
        #print('ASDFSDFASD')
        user_obj = User(
            user_id =  user_id,
            username =  username,
            first_name = first_name,
            # last_name = last_name,
            email =  email,
            is_staff = is_staff,
            # user_address =  user_address,
            date_joined =  date_joined,
            updated_at =  updated_at,
            is_deleted =  is_deleted,
            is_active =  is_active,
            is_superuser =  is_superuser,
            role_name =  role_name
            # phone_number =  phone_number
        )
        user_obj.set_password(data.get('password',None))
        #print("date_joined",date_joined,"\nupdated_at",updated_at)
        user_obj.save()
        resp = "User Account added successfully."
        status_code = 200
        return resp,status_code
    except Exception as e:
        status_code = 400
        resp = "User account could not be created. "+str(e)
        return resp,status_code



def get_user_account_util(user_id):
    """
    Usage: API to get details of a user account.
    Request Parameters: user_id
    Request Method: GET
    Response: {
        "password": "pbkdf2_sha256$180000$zDVi3fZKMmWY$aKuAzjTsK29jYke5iYTon6KT0sRPxS8XL6juzXx8vH4=",
        "last_login": null,
        "is_staff": false,
        "first_name": "Shivani",
        "last_name": "Udupa",
        "username": "shivu232",
        "email": "shivani@gmail.com",
        "user_address": "Bangalore",
        "date_joined": "2020-08-03T18:45:19.428Z",
        "updated_at": "2020-08-03T18:45:19.428Z",
        "is_deleted": false,
        "is_active": true,
        "is_superuser": false,
        "role_name": "Sysadmin",
        "phone_number": "21218739283",
        "groups": [],
        "user_permissions": []
    }
    """
    # resp = {
    #     "password": "pbkdf2_sha256$180000$zDVi3fZKMmWY$aKuAzjTsK29jYke5iYTon6KT0sRPxS8XL6juzXx8vH4=",
    #     "last_login": None,
    #     "is_staff": False,
    #     "first_name": "Shivani",
    #     "last_name": "Udupa",
    #     "username": "shivu232",
    #     "email": "shivani@gmail.com",
    #     "user_address": "Bangalore",
    #     "date_joined": "2020-08-03T18:45:19.428Z",
    #     "updated_at": "2020-08-03T18:45:19.428Z",
    #     "is_deleted": False,
    #     "is_active": True,
    #     "is_superuser": False,
    #     "role_name": "Sysadmin",
    #     "phone_number": "21218739283",
    #     "groups": [],
    #     "user_permissions": []
    # }

    # resp = {}
    try:
        user_obj = User.objects.filter(user_id=user_id)
        user_obj_json = json.loads(serializers.serialize('json',user_obj))
        resp = user_obj_json[0]['fields']
        return resp
    except Exception as e:
        resp = "User account could not be retrieved. "+str(e)
        return resp


def update_user_account_util(data):   #Note: Unable to update email or username as they are having unique constraints.
    """
    Usage: API to update a user account.
    Request Parameters: {
	    "user_id" : "2ca8ed2e-9675-444a-9ce8-5c01b8b445c5",
	    "first_name" : "Shruti",
	    "last_name" : "Udupa",
	    "is_staff" : false,
	    "user_address" : "Hyderabad",
	    "is_active" : true,
	    "is_superuser" : false,
	    "role_name" : "Manager",
	    "phone_number" : "7149383442"
    }
    Request Method: PATCH
    Response: {
        "message": "User Account updated successfully."
    }
    """
    resp = {}
    print(data)
    try:
        email = data.get('email',None)
        first_name = data.get('first_name',None)
        last_name = data.get('last_name',None)
        is_staff = data.get('is_staff',None)
        user_address = data.get('user_address',None)
        updated_at = timezone.now()
        is_active = data.get('is_active',None)
        is_superuser = data.get('is_superuser',None)
        role_name = data.get('role_name',None)
        phone_number = data.get('phone_number',None)
        user_obj = User.objects.get(email=email)
        print(email,'hhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhh')
        if user_obj:
            if first_name :
                user_obj.first_name=first_name
            if last_name :
                user_obj.last_name=last_name
            if is_staff :
                user_obj.is_staff=is_staff
            if updated_at :
                user_obj.updated_at=updated_at
            if is_active :
                user_obj.is_active=is_active
            if is_superuser :
                user_obj.is_superuser=is_superuser
            if role_name :
                user_obj.role_name=role_name
            user_obj.save()
            resp = "User Account updated successfully."
            return resp
        else:
            resp = "User account could not be updated. User ID is not specified."
            return resp   
    except Exception as e:
        resp = "User account could not be updated. "+str(e)
        return resp


def delete_user_account_util(username):
    """
    Usage: API to delete a user account.
    Request Parameters: user_id
    Request Method: DELETE
    Response: {
        "message": "User Account deleted successfully."
    }
    """
    resp = {}
    try:
        user_obj = User.objects.get(username=username)
        if user_obj:
            user_obj.is_deleted=True
            user_obj.save()
            resp = "User account deleted successfully."
            return resp
        else:
            resp = "User account could not be deleted. User ID is not specified."
            return resp 
    except Exception as e:
        resp = "User account could not be deleted. "+str(e)
        return resp


def get_all_user_accounts_util():
    """
    Usage: API to get all the existing user accounts.
    Request Parameters: None
    Request Method: GET
    Response: List of all user accounts.
    [{
        "password": "pbkdf2_sha256$180000$CAOl1CnTxpxe$fRXKkLU4iVGa4DwTW8naoKcDSqvLEKcS+sfNdryl8GM=",
        "last_login": null,
        "is_staff": true,
        "first_name": "Shruti",
        "last_name": "Udupa",
        "username": "shruti",
        "email": "shruti.udupa@lincode.us",
        "user_address": "Hyderabad",
        "date_joined": "2020-08-03T17:40:03.795Z",
        "updated_at": "2020-08-03T20:43:29.827Z",
        "is_deleted": false,
        "is_active": true,
        "is_superuser": true,
        "role_name": "Manager",
        "phone_number": "7149383442",
        "groups": [],
        "user_permissions": []
    }]
    """
    resp = []
    try:
        User_obj = User.objects.filter(is_deleted=False)
        for i in range(len(User_obj)):
            User_obj_json = json.loads(serializers.serialize('json',User_obj))
            User_obj_json = User_obj_json[i]['fields']
            resp.append(User_obj_json)
        return resp
    except Exception as e:
        resp = "Could not retrieve all the existing client accounts. "+str(e)
        return resp


################################################################USER_CLIENT CRUDS################################################################
def add_user_client_util(data):
    """
    Usage: API to add a user client account.
    Request Parameters: {
	    "username" : "client_admin_poo1",
	    "first_name" : "Pooja",
	    "last_name" : "Kandhal",
	    "email" : "poo@gmail.com",
	    "is_staff" : false,
	    "user_address" : "Mumbai",
        "is_superuser" : false,
        "role_name" : "438428eh32e",
        "phone_number" : "48372924366",
        "password" : "password",
	    "client_user_manager" : "None",
	    "is_client_admin" : true
    }
    Request Method: POST
    Response: {
        "message": "User Client Account added successfully."
    }
    """
    resp = {}
    try:
        user_id = str(uuid.uuid4()) 
        client_id = data.get('client_id',None)
        username = data.get('username',None)
        first_name = data.get('first_name',None)
        last_name = data.get('last_name',None)
        is_staff = data.get('is_staff',None)
        email = data.get('email',None)
        user_address = data.get('user_address',None)
        date_joined = timezone.now()
        updated_at = timezone.now()
        is_deleted = False
        is_active = True
        is_superuser = data.get('is_superuser',None)
        role_name = data.get('role_name',None)
        phone_number = data.get('phone_number',None)
        client_user_manager = data.get('client_user_manager',None)
        is_client_admin = data.get('is_client_admin',None)
        user_client_obj = User_Client(
            user_id =  user_id,
            client_id =  client_id,
            username =  username,
            first_name =  first_name,
            last_name =  last_name,
            is_staff = is_staff,
            email =  email,
            user_address =  user_address,
            date_joined =  date_joined,
            updated_at =  updated_at,
            is_deleted =  is_deleted,
            is_active =  is_active,
            is_superuser =  is_superuser,
            role_name =  role_name,
            phone_number =  phone_number,
            client_user_manager =  client_user_manager,
            is_client_admin = is_client_admin
        )
        user_client_obj.set_password(data.get('password',None))
        user_client_obj.save()
        resp = "User Client Account added successfully."
    
        return resp
    except Exception as e:
        resp = "User Client account could not be created. "+str(e)
        return resp

def update_user_client_util(data):
    """
    Usage: API to update a user client account.
    Request Parameters: {
	    "first_name" : "Pooja",
	    "last_name" : "Kandhal",
	    "is_staff" : false,
	    "user_address" : "Bombay",
	    "is_active" : true,
	    "is_superuser" : false,
	    "role_name" : "Manager",
	    "phone_number" : "7149383442",
	    "client_user_manager": "None", 
	    "is_client_admin": true
    }
    Request Method: PATCH
    Response: {
        "message": "User client Account updated successfully."
    }
    """
    resp = {}
    try:
        user_id = data.get('user_id',None)
        first_name = data.get('first_name',None)
        last_name = data.get('last_name',None)
        is_staff = data.get('is_staff',None)
        # user_address = data.get('user_address',None)
        updated_at = datetime.utcnow()
        is_active = data.get('is_active',None)
        is_superuser = data.get('is_superuser',None)
        role_name = data.get('role_name',None)
        # phone_number = data.get('phone_number',None)
        client_user_manager = data.get('client_user_manager',None)
        is_client_admin = data.get('is_client_admin',None)
        user_client_obj = User.objects.get(last_name=last_name)
        if user_client_obj:
            if first_name :
                user_client_obj.first_name=first_name
            if last_name :
                user_client_obj.last_name=last_name
            if is_staff :
                user_client_obj.is_staff=is_staff
            if updated_at :
                user_client_obj.updated_at=updated_at
            if is_active :
                user_client_obj.is_active=is_active
            if is_superuser :
                user_client_obj.is_superuser=is_superuser
            if role_name :
                user_client_obj.role_name=role_name
            if client_user_manager:
                user_client_obj.client_user_manager=client_user_manager
            if client_user_manager:
                user_client_obj.is_client_admin=is_client_admin
            user_client_obj.save()
            resp = "User Client Account updated successfully."
            return resp
        else:
            resp = "User Client account could not be updated. Client ID is not specified."
            return resp   
    except Exception as e:
        resp = "User Client account could not be updated. "+str(e)
        return resp

def delete_user_client_util(user_id):
    resp = {}
    try:
        user_client_obj = User_Client.objects.get(user_id=user_id)
        if user_client_obj:
            user_client_obj.is_deleted=True
            user_client_obj.save()
            resp = "User Client account deleted successfully."
            return resp
        else:
            resp = "User Client account could not be deleted. User ID is not specified."
            return resp 
    except Exception as e:
        resp = "User Client account could not be deleted. "+str(e)
        return resp

def get_user_client_util(user_id):
    """
    Usage: API to get details of a user client account.
    Request Parameters: user_id
    Request Method: GET
    Response: {
        "user_id": "a83b9fc4-8171-43aa-927f-487feac73ae3",
        "client_user_manager": "None",
        "is_client_admin": true
    }
    """
    resp = {}
    try:
        user_client_obj = User_Client.objects.filter(user_id=user_id)
        user_client_obj_json = json.loads(serializers.serialize('json',user_client_obj))
        resp = user_client_obj_json[0]['fields']
        return resp
    except Exception as e:
        resp = "User client account could not be retrieved. "+str(e)
        return resp

def get_user_clients_util():
    """
    Usage: API to get all the existing user accounts.
    Request Parameters: None
    Request Method: GET
    Response: List of all user accounts.
    [{
        "user_id": "a83b9fc4-8171-43aa-927f-487feac73ae3",
        "client_user_manager": "None",
        "is_client_admin": true
    }]
    """
    resp = []
    try:
        user_client_obj = User_Client.objects.filter(is_deleted=False)
        for i in range(len(user_client_obj)):
            user_client_obj_json = json.loads(serializers.serialize('json',user_client_obj))
            user_client_obj_json = user_client_obj_json[i]['fields']
            resp.append(user_client_obj_json)
        return resp
    except Exception as e:
        resp = "Could not retrieve all the existing user client accounts. "+str(e)
        return resp


#########################################################USER_SI CRUDS########################################################
def add_user_si_util(data):
    """
    Usage: API to add a user SI account.
    Request Parameters: {
	    "username" : "client_admin_poo1",
	    "first_name" : "Pooja",
	    "last_name" : "Kandhal",
	    "email" : "poo@gmail.com",
	    "is_staff" : false,
	    "user_address" : "Mumbai",
        "is_superuser" : false,
        "role_name" : "438428eh32e",
        "phone_number" : "48372924366",
        "password" : "password",
    }
    Request Method: POST
    Response: {
        "message": "User SI Account added successfully."
    }
    """
    resp = {}
    try:
        user_id = str(uuid.uuid4()) 
        si_id = data.get('si_id',None)
        username = data.get('username',None)
        first_name = data.get('first_name',None)
        last_name = data.get('last_name',None)
        is_staff = data.get('is_staff',None)
        email = data.get('email',None)
        user_address = data.get('user_address',None)
        date_joined = timezone.now()
        updated_at = timezone.now()
        is_deleted = False
        is_active = True
        is_superuser = data.get('is_superuser',None)
        role_name = data.get('role_name',None)
        phone_number = data.get('phone_number',None)
        user_si_obj = User_SI(
            user_id =  user_id,
            si_id =  si_id,
            first_name =  first_name,
            last_name =  last_name,
            username =  username,
            email =  email,
            user_address =  user_address,
            date_joined =  date_joined,
            updated_at =  updated_at,
            is_deleted =  is_deleted,
            is_active =  is_active,
            is_staff =  is_staff,
            is_superuser =  is_superuser,
            role_name =  role_name,
            phone_number =  phone_number
        )
        user_si_obj.set_password(data.get('password',None))
        user_si_obj.save()
        resp = "User SI Account added successfully."
    
        return resp
    except Exception as e:
        resp = "User SI account could not be created. "+str(e)
        return resp

def update_user_si_util(data):
    """
    Usage: API to update a user SI account.
    Request Parameters: {
	    "user_id": "bb98f0bb-a33e-4143-9885-5afc5342ba1d",
	    "first_name" : "Jinta",
	    "last_name" : "Mary Skariah",
	    "is_staff" : false,
	    "user_address" : "Alapuzzha",
	    "is_active" : true,
	    "is_superuser" : false,
	    "role_name" : "Manager",
	    "phone_number" : "7149383442"
    }
    Request Method: PATCH
    Response: {
        "message": "User SI Account updated successfully."
    }
    """
    resp = {}
    try:
        user_id = data.get('user_id',None)
        first_name = data.get('first_name',None)
        last_name = data.get('last_name',None)
        user_address = data.get('user_address',None)
        updated_at = timezone.now()
        is_active = data.get('is_active',None)
        is_staff = data.get('is_staff',None)
        is_superuser = data.get('is_superuser',None)
        #role_name = data.get('role_name',None)
        #phone_number = data.get('phone_number',None)
        user_si_obj = User_SI.objects.get(user_id=user_id)
        #print("-----user_si_obj: ",user_si_obj.role_name)
        if user_si_obj:
            if first_name :
                user_si_obj.first_name=first_name
            if last_name :
                user_si_obj.last_name=last_name
            if user_address :
                user_si_obj.user_address=user_address
            if updated_at :
                user_si_obj.updated_at=updated_at
            if is_active :
                user_si_obj.is_active=is_active
            if is_staff :
                user_si_obj = user_si_obj.is_staff=is_staff
            if is_superuser :
                user_si_obj = user_si_obj.is_superuser=is_superuser
            #if role_name :
                #user_si_obj = user_si_obj.role_name=role_name    unknown error while updating the role and phonenumber
            #if phone_number :
                #user_si_obj = user_si_obj.phone_number=phone_number
            user_si_obj.save()
            resp = "User SI Account updated successfully."
            return resp
        else:
            resp = "User SI account could not be updated. SI ID is not specified."
            return resp   
    except Exception as e:
        resp = "User SI account could not be updated. "+str(e)
        return resp

def delete_user_si_util(user_id):
    resp = {}
    try:
        User_si_obj = User_SI.objects.get(user_id=user_id)
        if User_si_obj:
            User_si_obj.is_deleted=True
            User_si_obj.save()
            resp = "User SI account deleted successfully."
            return resp
        else:
            resp = "User SI account could not be deleted. SI ID is not specified."
            return resp 
    except Exception as e:
        resp = "User SI account could not be deleted. "+str(e)
        return resp

def get_user_si_util(user_id):
    resp = {}
    try:
        User_si_obj = User_SI.objects.filter(user_id=user_id)
        User_si_obj_json = json.loads(serializers.serialize('json',User_si_obj))
        resp = User_si_obj_json[0]['fields']
        return resp
    except Exception as e:
        resp = "User SI account could not be retrieved. "+str(e)
        return resp

def get_user_sis_util():
    resp = []
    try:
        User_si_obj = User_SI.objects.filter(is_deleted=False)
        for i in range(len(User_si_obj)):
            User_si_obj_json = json.loads(serializers.serialize('json',User_si_obj))
            User_si_obj_json = User_si_obj_json[i]['fields']
            resp.append(User_si_obj_json)
        return resp
    except Exception as e:
        resp = "Could not retrieve all the existing user SI accounts. "+str(e)
        return resp


################################################################CLIENT CRUDS################################################################

def add_client_account_util(data):
    """
    Usage: API to add a client account.
    Request Parameters: {
	    "client_name" : "Cataler",
	    "client_license_key" : "sdwej9883",
	    "client_role" : "Role1",
	    "assigned_to" : "Client_User1",
	    "client_address" : "Chennai"
    }
    Request Method: POST
    Response: {
        "message": "Client Account added successfully."
    }
    """
    resp = {}
    try:
        client_id = str(uuid.uuid4())
        client_name = data.get('client_name',None)
        client_license_key = data.get('client_license_key',None)
        client_role = data.get('client_role',None)
        assigned_to = data.get('assigned_to',None)
        client_address = data.get('client_address',None)

        is_deleted = False

        client_obj = Client(
            client_id = client_id,
            client_name = client_name,
            client_license_key = client_license_key,
            client_role = client_role,
            assigned_to = assigned_to,
            client_address = client_address,
            is_deleted = is_deleted
        )
        client_obj.save()
        resp = "Client Account added successfully."
        return resp
    except Exception as e:
        resp = "Client account could not be created. "+str(e)
        return resp


def get_client_account_util(client_id):
    """
    Usage: API to get details of a client account.
    Request Parameters: client_id
    Request Method: GET
    Response: {
        "client_id": "009aaf6d-bc78-41b0-ae97-3fd20e6d1874",
        "client_name": "Client3",
        "client_license_key": "sdwej9883",
        "client_role": "manager",
        "assigned_to": "User2",
        "client_address": "Chennai",
        "is_deleted": false

    }
    """
    resp = {}
    try:
        client_obj = Client.objects.filter(client_id=client_id)
        client_obj_json = json.loads(serializers.serialize('json',client_obj))
        resp = client_obj_json[0]['fields']
        return resp
    except Exception as e:
        resp = "Client account could not be retrieved. "+str(e)
        return resp


def update_client_account_util(data):
    """
    Usage: API to update a client account.
    Request Parameters: {
        "client_id": "009aaf6d-bc78-41b0-ae97-3fd20e6d1874"
	    "client_name" : "Client3",
	    "client_license_key" : "sdwej9883",
	    "client_role" : "manager",
	    "assigned_to" : "User2",
	    "client_address" : "Chennai"
    }
    Request Method: PATCH
    Response: {
        "message": "Client Account updated successfully."
    }
    """
    resp = {}
    try:
        client_id = data.get('client_id',None)
        client_name = data.get('client_name',None)
        client_license_key = data.get('client_license_key',None)
        client_role = data.get('client_role',None)
        assigned_to = data.get('assigned_to',None)
        client_address = data.get('client_address',None)
        client_obj = Client.objects.filter(client_id=client_id)
        if client_obj:
            if client_name:
                client_obj.update(client_name=client_name)
            if client_license_key:
                client_obj.update(client_license_key=client_license_key)
            if client_role:
                client_obj.update(client_role=client_role)
            if assigned_to:
                client_obj.update(assigned_to=assigned_to)
            if client_address:
                client_obj.update(client_address=client_address)
            resp = "Client Account updated successfully."
            return resp
        else:
            resp = "Client account could not be updated. Client ID is not specified."
            return resp   
    except Exception as e:
        resp = "Client account could not be updated. "+str(e)
        return resp


def delete_client_account_util(client_id):
    """
    Usage: API to delete a client account.
    Request Parameters: client_id
    Request Method: DELETE
    Response: {
        "message": "Client Account deleted successfully."
    }
    """
    resp = {}
    try:
        client_obj = Client.objects.filter(client_id=client_id)
        if client_obj:
            client_obj.update(is_deleted=True)

            resp = "Client account deleted successfully."
            return resp
        else:
            resp = "Client account could not be deleted. Client ID is not specified."
            return resp 
    except Exception as e:
        resp = "Client account could not be deleted. "+str(e)
        return resp


def get_all_client_accounts_util():
    """
    Usage: API to get all the existing client accounts.
    Request Parameters: None
    Request Method: GET
    Response: List of all client accounts.
    [{
        "client_id": "c2",
        "client_name": "Client2",
        "client_license_key": "sdwej9883",
        "client_role": "manager",
        "assigned_to": "User2",
        "client_address": "Chennai",
        "is_deleted": false
    }]
    """
    resp = []
    try:
        client_obj = Client.objects.filter(is_deleted=False)
        for i in range(len(client_obj)):
            client_obj_json = json.loads(serializers.serialize('json',client_obj))
            client_obj_json = client_obj_json[i]['fields']
            resp.append(client_obj_json)
        return resp
    except Exception as e:
        resp = "Could not retrieve all the existing client accounts. "+str(e)
        return resp


################################################################SI CRUDS################################################################
def add_si_account_util(data):
    """
    Usage: API to add a client account.
    Request Parameters: {
        "client_name" : "Client3",
        "client_license_key" : "sdwej9883",
        "client_role" : "manager",
	    "assigned_to" : "User2",
	    "client_address" : "Chennai",
        "activeClients" : "dewqdewq",
        "license_key_list" : "qewerade"
    }
    Request Method: POST
    Response: {
        "message": "SI Account added successfully."
    }
    """
    resp = {}
    try:
        client_id = str(uuid.uuid4())
        client_name = data.get('client_name',None)
        client_license_key = data.get('client_license_key',None)
        client_role = data.get('client_role',None)
        assigned_to = data.get('assigned_to',None)
        client_address = data.get('client_address',None)
        is_deleted = False
        activeClients = data.get('activeClients',None)
        license_key_list = data.get('license_key_list',None)
        si_obj = SI(
            client_id = client_id,
            client_name = client_name,
            client_license_key = client_license_key,
            client_role = client_role,
            assigned_to = assigned_to,
            client_address = client_address,
            is_deleted = is_deleted,
            activeClients = activeClients,
            license_key_list = license_key_list
            )
        si_obj.save()
        resp = "SI Account added successfully."
        return resp
    except Exception as e:
        resp = "SI account could not be created. "+str(e)
        return resp


def get_si_account_util(client_id):
    """
    Usage: API to get details of a client account.
    Request Parameters: client_id
    Request Method: GET
    Response: {
        "client_ptr": "2fdafd13-1839-4484-a705-a8c85e2852b5",
        "activeClients": "Cataler",
        "license_key_list": "qewerade"
    }
    """
    resp = {}
    try:
        si_obj = SI.objects.filter(client_id=client_id)
        si_obj_json = json.loads(serializers.serialize('json',si_obj))
        resp = si_obj_json[0]['fields']
        return resp
    except Exception as e:
        resp = "SI account could not be retrieved. "+str(e)
        return resp


def update_si_account_util(data):
    """
    Usage: API to update a client account.
    Request Parameters: {
	    "client_name" : "Client3",
	    "client_license_key" : "sdwej9883",
	    "client_role" : "manager",
	    "assigned_to" : "User2",
	    "client_address" : "Chennai",
        "client_id": "hdeyr734-3437-sde-r34f-wfwrr423d34",
        "activeClients" : "dewqdewq",
        "license_key_list" : "qewerade"
    }
    Request Method: PATCH
    Response: {
        "message": "Client Account updated successfully."
    }
    """
    resp = {}
    try:
        client_name = data.get('client_name',None)
        client_license_key = data.get('client_license_key',None)
        client_role = data.get('client_role',None)
        assigned_to = data.get('assigned_to',None)
        client_address = data.get('client_address',None)
        client_id = data.get('client_id',None)
        activeClients = data.get('activeClients',None)
        license_key_list = data.get('license_key_list',None)
        si_obj = SI.objects.get(client_id=client_id)
        if si_obj:
            if client_name:
                si_obj.client_name=client_name
            if client_license_key:
                si_obj.client_license_key=client_license_key
            if client_role:
                si_obj.client_role=client_role
            if assigned_to:
                si_obj.assigned_to=assigned_to
            if client_address:
                si_obj.client_address=client_address
            if activeClients:
                si_obj.activeClients=activeClients
            if license_key_list:
                si_obj.license_key_list=license_key_list
            si_obj.save()
            resp = "SI Account updated successfully."
            return resp
        else:
            resp = "SI account could not be updated. SI ID is not specified."
            return resp   
    except Exception as e:
        resp = "SI account could not be updated. "+str(e)
        return resp


def delete_si_account_util(client_id):
    """
    Usage: API to delete a SI account.
    Request Parameters: client_id
    Request Method: DELETE
    Response: {
        "message": "SI Account deleted successfully."
    }
    """
    resp = {}
    try:
        si_obj = SI.objects.get(client_id=client_id)
        if si_obj:
            si_obj.is_deleted=True
            si_obj.save()
            resp = "SI account deleted successfully."
            return resp
        else:
            resp = "SI account could not be deleted. SI ID is not specified."
            return resp 
    except Exception as e:
        resp = "SI account could not be deleted. "+str(e)
        return resp


def get_all_si_accounts_util():
    """
    Usage: API to get all the existing SI accounts.
    Request Parameters: None
    Request Method: GET
    Response: List of all SI accounts.
    [{
        "client_ptr": "2fdafd13-1839-4484-a705-a8c85e2852b5",
        "activeClients": "Cataler1",
        "license_key_list": "je822112"
    }]
    """
    resp = []
    try:
        si_obj = SI.objects.filter(is_deleted=False)
        for i in range(len(si_obj)):
            si_obj_json = json.loads(serializers.serialize('json',si_obj))
            si_obj_json = si_obj_json[i]['fields']
            resp.append(si_obj_json)
        return resp
    except Exception as e:
        resp = "Could not retrieve all the existing SI accounts. "+str(e)
        return resp


###########################################################User based Authentication APIS###################################################
  
def login_user_util(data):
    message = ""
    email = data.get('email',None)
    password = data.get('password',None)
    #workstation_id = data.get('workstation_name',None)
    #print("workstation_id::::::",workstation_id)
    #if not email or not password or not workstation_id:
    #    return "MissingField"
    #else:

    try:
        mp = MongoHelper().getCollection(WORKSTATION_COLLECTION)
    except:
        message = "Cannot connect to db"
        status_code = 500
        return message,status_code

    p = [p for p in mp.find()]

    workstation_id = p[0]['_id']
    # workstation_id = '123'
    try:
        user = authenticate(email=email, password=password)
        if user:
            #print("user::::::",user)
            user_object = {str(i.name):getattr(user, i.name) for i in user._meta.fields}
            #print("user_object::::::",user_object)
            del user_object['password']
            user_object['workstation_id'] = str(workstation_id)
            #cc = CacheHelper()
            #user_id = user_object['user_id']
            #print("user_id::::::",user_id)
            #user_id_key = RedisKeyBuilderServer(workstation_id).get_key(0, 'user_id_key')
            #print("user_id_key::::::",user_id_key)
            #cc.set_json({user_id_key : user_id}) 
            return user_object
        else:
            return "AuthError"
    except Exception as e:
        print(e)
        return "AuthError"
        


def login_supervisor_util(data):
    message = ""
    email = data.get('email',None)
    password = data.get('password',None)
    #workstation_name = data.get('workstation_name',None)
    #if not email or not password or not workstation_name:
    #    return "MissingField"
    #else:
    try:
        user = authenticate(email=email, password=password)
        if user:
            if user.role_name == "Supervisor":
                user_obj = {
                    'user_id' : user.user_id,
                    'first_name' : user.first_name,
                    'last_name' : user.last_name,
                    'username' : user.username,
                    'role_name' : user.role_name,
                    'email' : user.email,
                    'user_address' : user.user_address,
                    'date_joined' : str(user.date_joined),
                    'updated_at' : str(user.updated_at),
                    'is_active' : user.is_active,
                    'is_superuser' : user.is_superuser,
                    'phone_number' : user.phone_number,
                }
                return user_obj
            else:
                return "AuthError"
        else:
            return "AuthError"
    except Exception as e:
        return str(e)



'''def logout_user_util(data):
    if request.method == 'POST':
    	logout(data)
    	return HttpResponse(json.dumps({'redirect': '/'}), content_type="application/json")
    else:
    	return HttpResponseBadRequest(json.dumps({'error': 'Method type not allowed'}), content_type='application/json')
'''

