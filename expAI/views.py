import zipfile
import os
import datetime
import uuid
from rest_framework import viewsets, status
from django.db.models import Count
from expAI.utils import gen_report_docx
from .models import *
from .serializers import *
from rest_framework import views, generics, response, permissions, authentication
from rest_framework.response import Response
from rest_framework.decorators import action
from drf_yasg.utils import swagger_auto_schema
from drf_yasg import openapi
from django.contrib.auth import login, logout
from django.conf import settings
from rest_framework.permissions import IsAuthenticated
from .permissions import *
from .paginations import *
from rest_framework import filters
from .filters import *
from django_filters.rest_framework import DjangoFilterBackend
from django.utils.decorators import method_decorator
from rest_framework.parsers import FileUploadParser, FormParser, MultiPartParser
from rest_framework.parsers import FileUploadParser, FormParser, MultiPartParser
from .AI import *
from .AI_models.Pytorch_Retinaface import Retina_mnet
from .AI_models.Pytorch_FaceNet import FaceNet
from .AI_models.Tensorflow_FightDetecttion import fightdetection
from django.http import JsonResponse
from django.shortcuts import get_object_or_404
# Create your views here.


class CsrfExemptSessionAuthentication(authentication.SessionAuthentication):
    def enforce_csrf(self, request):
        return


class expAIViewSet(viewsets.ModelViewSet):
    """
    This viewset automatically provides `list`, `create`, `retrieve`,
    `update` and `destroy` actions.

    Additionally we also provide an extra `checkBody` action.
    """
    queryset = Softwarelibs.objects.all()
    serializer_class = SoftwareLibsSerializer
    authentication_classes = (CsrfExemptSessionAuthentication,)

@method_decorator(name="list", decorator=swagger_auto_schema(manual_parameters=[openapi.Parameter('datasetName', openapi.IN_QUERY, description='Tên bộ dữ liệu', type=openapi.TYPE_STRING), openapi.Parameter('datasetSumFrom', openapi.IN_QUERY, description='Cận dưới số lượng', type=openapi.TYPE_INTEGER),
                                                                                openapi.Parameter(
                                                                                    'datasetSumTo', openapi.IN_QUERY, description='Cận trên số lượng', type=openapi.TYPE_INTEGER),
                                                                                openapi.Parameter('datasetOwner', openapi.IN_QUERY,
                                                                                                  description='ID người tạo', type=openapi.TYPE_INTEGER),
                                                                                openapi.Parameter('datasetProb', openapi.IN_QUERY, description='Bài toán áp dụng', type=openapi.TYPE_INTEGER)]))
class DatasetsViewSet(viewsets.ModelViewSet):
    """
    This viewset automatically provides `list`, `create`, `retrieve`,
    `update` and `destroy` actions.

    Additionally we also provide an extra `checkBody` action.
    """
    authentication_classes = (CsrfExemptSessionAuthentication,)
    serializer_class = DatasetsSerializer
    # permission_classes = [IsOwner | IsAdmin]
    pagination_class = LargeResultsSetPagination
    datasetname = openapi.Parameter(
        'datasetname', openapi.IN_QUERY, description='Tên bộ dữ liệu', type=openapi.TYPE_STRING)
    # filter_backends = [DatasetsFilter]
    # search_fields = ['datasetname', 'datasetfolderurl','datasettraining','datasettesting','datasetsum','datasetcreator','datasetdescription']
    # filter_fields = ['datasetname', 'datasetfolderurl','datasettraining','datasettesting','datasetsum','datasetcreator','datasetdescription']
    def create(self, request, *args, **kwargs):
        usr = self.request.user
        if (usr.roleid.rolename == "STUDENT") and (request.data["datasettype"]==2):
            return Response("Học viên không được tạo public")
        request.data["datasetowner"]=usr.id
        serializer = self.get_serializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        self.perform_create(serializer)
        headers = self.get_success_headers(serializer.data)
        return Response(serializer.data, status=status.HTTP_201_CREATED, headers=headers)

    def get_queryset(self):
        print("a")
        usr = self.request.user
        try:
            usr = User.objects.get(email=usr.email)
        except:
            print("User not authorized") 
            return
        print("b")
        a = ClassUser.objects.filter(status=1).filter(user_id=usr.id)
        b = list(a.values_list('class_id', flat=True))
        c = sum([list(ClassUser.objects.filter(status=1).filter(
            class_id=i).values_list("user_id", flat=True)) for i in b], [])
        d = User.objects.filter(is_active=1, id__in=c)
        print(a, b, c, d)
        if usr.roleid.rolename == "ADMIN":
            queryset = Datasets.objects.all()
            
        elif usr.roleid.rolename == "STUDENT":
            queryset = Datasets.objects.filter(
                datasettype=1) | Datasets.objects.filter(datasetowner=self.request.user)
        else:  # teacher

            queryset = Datasets.objects.filter(
                datasettype=1) | Datasets.objects.filter(datasetowner__in=d)
        datasetname = self.request.query_params.get('datasetname')
        datasetProb = self.request.query_params.get('datasetProb')
        datasetSumTo = self.request.query_params.get('datasetSumTo')
        datasetSumFrom = self.request.query_params.get('datasetSumFrom')
        datasetOwner = self.request.query_params.get('datasetOwner')
        queryset = queryset.filter(
            datasetsum__lte=datasetSumTo) if datasetSumTo != None else queryset
        queryset = queryset.filter(
            datasetsum__gte=datasetSumFrom) if datasetSumFrom != None else queryset
        queryset = queryset.filter(
            datasetproblem=datasetProb) if datasetProb != None else queryset
        queryset = queryset.filter(
            datasetowner=datasetOwner) if datasetOwner != None else queryset
        queryset = queryset.filter(
            datasetname__icontains=datasetname) if datasetname != None else queryset
        return queryset

    # def perform_create(self, serializer):
    #     serializer.save(datasetowner=self.request.user)

    def update(self, request, *args, **kwargs):
        partial = kwargs.pop('partial', False)
        usr = self.request.user
        instance = self.get_object()
        # if usr.roleid.rolename != "ADMIN" or int(usr.id) != int(instance.datasetowner.id):
        #     return Response("Tài khoản cần Admin hoặc sở hữu dataset này",status=status.HTTP_401_UNAUTHORIZED)
        serializer = self.get_serializer(instance, data=request.data, partial=partial)
        serializer.is_valid(raise_exception=True)
        self.perform_update(serializer)

        if getattr(instance, '_prefetched_objects_cache', None):
            # If 'prefetch_related' has been applied to a queryset, we need to
            # forcibly invalidate the prefetch cache on the instance.
            instance._prefetched_objects_cache = {}

        return Response(serializer.data)
    def destroy(self, request, *args, **kwargs):
        try:
            usr = self.request.user
            print(usr.roleid.rolename)
            instance = self.get_object()
            # if usr.roleid.rolename != "ADMIN" and int(usr.id) != int(instance.datasetowner.id):
            #     return Response("Tài khoản cần Admin hoặc sở hữu dataset này",status=status.HTTP_401_UNAUTHORIZED)
            try:
                import shutil
                shutil.rmtree(f'datasets/{instance.datasetfolderurl}')
            except:...
            self.perform_destroy(instance)
        except:
            return Response(status=status.HTTP_500_INTERNAL_SERVER_ERROR)

        return Response(status=status.HTTP_204_NO_CONTENT)


class ClassesViewSet(viewsets.ModelViewSet):
    """
    This viewset automatically provides `list`, `create`, `retrieve`,
    `update` and `destroy` actions.

    Additionally we also provide an extra `checkBody` action.
    """
    # permission_classes = (IsAdmin,)
    queryset = Class.objects.all()
    pagination_class = LargeResultsSetPagination
    serializer_class = ClassesSerializer
    authentication_classes = (CsrfExemptSessionAuthentication,)

class AccountsViewSet(viewsets.ModelViewSet):
    """
    This viewset automatically provides `list`, `create`, `retrieve`,
    `update` and `destroy` actions.

    Additionally we also provide an extra `checkBody` action.
    """
    # permission_classes = (IsAdmin,)
    queryset = User.objects.all()
    pagination_class = LargeResultsSetPagination
    serializer_class = User2Serializer
    filter_backends = [filters.SearchFilter]
    search_fields = ['email', 'name']
    authentication_classes = (CsrfExemptSessionAuthentication,)
    def get_serializer_class(self):
        if self.request.method == 'POST':
            return UserCreateSerializer
        return User2Serializer
    def perform_destroy(self, instance):
        instance.is_active = False
        instance.save()
    def get_userclass_classname(self, user_id):
        classuser = ClassUser.objects.filter(user_id=user_id, status=1).first()
        if classuser:
            classname = classuser.class_id.classname
        else:
            classname = None
        return classname
    def update(self, request, *args, **kwargs):
        from django.contrib.auth.hashers import (
            make_password,
        )
        partial = kwargs.pop('partial', False)
        instance = self.get_object()
        # request.data["password"] = make_password(request.data["password"])
        serializer = self.get_serializer(
            instance, data=request.data, partial=partial)
        serializer.is_valid(raise_exception=True)
        self.perform_update(serializer)

        if getattr(instance, '_prefetched_objects_cache', None):
            # If 'prefetch_related' has been applied to a queryset, we need to
            # forcibly invalidate the prefetch cache on the instance.
            instance._prefetched_objects_cache = {}

        return Response(serializer.data)
    def retrieve(self, request, *args, **kwargs):
        instance = self.get_object()
        serializer = self.get_serializer(instance)
        classname = self.get_userclass_classname(instance.id)
        if classname:
            serializer.data['usrclass'] = classname
        return Response(serializer.data)

    # Override the list method to include userclass classname in the response
    def list(self, request, *args, **kwargs):
        queryset = self.filter_queryset(self.get_queryset())
        page = self.paginate_queryset(queryset)
        if page is not None:
            serializer = self.get_serializer(page, many=True)
            response_data = serializer.data
            for user_data in response_data:
                user_id = user_data['id']
                classname = self.get_userclass_classname(user_id)
                user_data['usrclass'] = classname
            return self.get_paginated_response(response_data)

        serializer = self.get_serializer(queryset, many=True)
        response_data = serializer.data
        for user_data in response_data:
            user_id = user_data['id']
            classname = self.get_userclass_classname(user_id)
            user_data['usrclass'] = classname
        return Response(response_data)
class ThongkeRole(generics.RetrieveAPIView):
    queryset= User.objects.all()
    serializer_class= ThongkeRole
    def get(self,request):
        roles = self.queryset.values('roleid').annotate(count=Count('roleid'))
        
        name_role=["admin","giaovien", "hocvien"]
        role_counts = {}
        for role in roles:
            if role['roleid'] == None: continue
            role_counts[name_role[int(role['roleid'])-1]] = role['count']

        return Response(role_counts)
class ChangeUserPasswordView(generics.UpdateAPIView):
    queryset = User.objects.all()
    serializer_class = ChangePassword2Serializer
    authentication_classes = (CsrfExemptSessionAuthentication,)
    def get_object(self, queryset=None):
        id_user = self.request.data.get('id_user')
        obj = self.queryset.get(id=id_user)
        return obj

    def update(self, request):
        """
        Change User's Password API
        """
        obj = self.get_object()
        new_password = request.data.get('new_password')
        obj.set_password(new_password)
        obj.save()
        return Response({"result": "Success"})


class DisableAccountView(generics.UpdateAPIView):
    queryset = User.objects.all()
    serializer_class = DisableAccountSerializer
    authentication_classes = (CsrfExemptSessionAuthentication,)
    def get_object(self, queryset=None):
        obj = self.request.user
        obj = User.objects.get(id=obj.pk)
        return obj

    def update(self, request):
        """
        Change User's Password API
        """
        obj = self.get_object()
        obj.is_active = 0
        obj.save()
        return Response({"result": "Success"})


class RequestToClassView(generics.CreateAPIView):
    queryset = User.objects.all()
    serializer_class = RequestToClassSerializer
    authentication_classes = (CsrfExemptSessionAuthentication,)
    def get_object(self, queryset=None):
        obj = self.request.user

        return obj

    def post(self, request):
        """
        Change User's Password API
        """
        usr = User.objects.get(id=self.request.data.get('id_user'))
        class_user = ClassUser(
            user_id=User.objects.get(id=self.request.data.get('id_user')),
            class_id = Class.objects.get(classid=self.request.data.get('id_class')),
            status = 0,
            time_regis = datetime.datetime.today().strftime('%Y-%m-%d'),
            time_approve = '2000-10-22',
        )
        usr.chose_class=True
        print(usr.chose_class, usr.email)
        usr.save()
        class_user.save()
        return Response({"result": "Success"})


class AssignTeacherToClassView(generics.CreateAPIView):
    queryset = ClassUser.objects.all()
    serializer_class = AssignToClassSerializer
    authentication_classes = (CsrfExemptSessionAuthentication,)
    def check_more_than_one(self, class_id):
        a=set(ClassUser.objects.filter(status=1).filter(class_id=class_id).values_list("user_id",flat=True))
        if len(User.objects.filter(is_active=1).filter(id__in=a))>0:
            return False
        else:
            return True
    def post(self, request):
        if not self.check_more_than_one(self.request.data.get('id_class')):
            return Response({"result": "Class already had teacher"})
        class_user = ClassUser(
            user_id=User.objects.get(id=self.request.data.get('id_user')),
            class_id = Class.objects.get(classid=self.request.data.get('id_class')),
            status = 1,
            time_regis = datetime.datetime.today().strftime('%Y-%m-%d'),
            time_approve = '2000-10-22',
        )
        class_user.save()
        return Response({"result": "Success"})


class ApproveToClassView(generics.UpdateAPIView):
    authentication_classes = (CsrfExemptSessionAuthentication,)
    queryset = ClassUser.objects.all()
    serializer_class = ApproveToClassSerializer
    def check_more_than_one(self, user_id):
        a = ClassUser.objects.filter(status=1).filter(user_id=user_id).values_list("class_id",flat=True)
        if len(set(a))> 0:
            return False
        else:
            return True

    def update(self, request):
        obj = ClassUser.objects.get(class_user_id= self.request.data.get('id_user_class'))
        if not self.check_more_than_one(obj.user_id):
            return Response({"result": "Học viên đẫ ở trong lớp"})
        obj.time_approve = datetime.datetime.today()
        obj.status = 1
        obj.save()
        obj2 = obj.user_id
        obj2.usrclass = obj.class_id.classname

        obj2.chose_class = True
        obj2.save()
        return Response({"result": "Success"})
class DenyToClassView(generics.UpdateAPIView):
    authentication_classes = (CsrfExemptSessionAuthentication,)
    queryset = ClassUser.objects.all()
    serializer_class = DenyToClassSerializer


    def update(self, request):
        obj = ClassUser.objects.get(class_user_id= self.request.data.get('id_user_class'))
        
        obj.time_approve = datetime.datetime.today()
        obj.status = 2
        obj.save()

        return Response({"result": "Success"})

class GetAllClassUserView(generics.ListAPIView):
    authentication_classes = (CsrfExemptSessionAuthentication,)
    queryset = ClassUser.objects.all()
    serializer_class = UserClass2Serializer
    pagination_class = LargeResultsSetPagination
class ApproveUserRequestView(generics.UpdateAPIView):
    queryset = User.objects.all()
    serializer_class = ConfirmUserSerializer
    authentication_classes = (CsrfExemptSessionAuthentication,)
    def update(self, request):
        obj = User.objects.get(id= self.request.data.get('id_user'))

        obj.is_active = self.request.data.get('status')
        obj.save()
        return Response({"result": "Success"})

class LoginView(generics.CreateAPIView):
    serializer_class = LoginSerializer
    permission_classes = (permissions.AllowAny,)
    authentication_classes = (CsrfExemptSessionAuthentication,)

    @swagger_auto_schema(tags=['Đăng nhập - Đăng ký'])
    def post(self, serializer):
        serializer = LoginSerializer(data=self.request.data)
        serializer.is_valid(raise_exception=True)
        user = serializer.validated_data['user']
        login(self.request, user)
        return response.Response(UserSerializer(user).data)


class LogoutView(views.APIView):
    authentication_classes = (CsrfExemptSessionAuthentication,)

    @swagger_auto_schema(tags=['Đăng nhập - Đăng ký'])
    def post(self, request):
        logout(request)
        return response.Response()


@method_decorator(name="post", decorator=swagger_auto_schema(tags=["Đăng nhập - Đăng ký"]))
class RegisterView(generics.CreateAPIView):
    serializer_class = UserSerializer
    permission_classes = (permissions.AllowAny,)

    @swagger_auto_schema(tags=['Đăng nhập - Đăng ký'])
    def perform_create(self, serializer):
        serializer.is_staff = False
        user = serializer.save()
        # user.backend = settings.AUTHENTICATION_BACKENDS[0]
        login(self.request, user)


class DeleteClassUserView(viewsets.ModelViewSet):
    authentication_classes = (CsrfExemptSessionAuthentication,)
    queryset = ClassUser.objects.all()
    serializer_class = UserClassSerializer
    permission_classes = [IsAdmin]
    filterset_fields = ['user_id','class_id', 'status','user_id__roleid__rolename']
    def get_serializer_class(self):
        if self.action == 'list':  # use the list serializer for list action
            return ClassUserSerializer
        return super().get_serializer_class()
    def get_queryset(self):
        if self.action == 'list':
            return ClassUser.objects.filter(status=1)
        return super().get_queryset()
class DeleteClassUserView(viewsets.ModelViewSet):
    authentication_classes = (CsrfExemptSessionAuthentication,)
    queryset = ClassUser.objects.all()
    serializer_class = UserClassSerializer
    permission_classes = [IsAdmin]
    pagination_class = LargeResultsSetPagination
    def check_more_than_one(self, class_id):
        a=set(ClassUser.objects.filter(status=1).filter(class_id=class_id).values_list("user_id",flat=True))
        if len(User.objects.filter(is_active=1).filter(id__in=a))>0:
            return False
        else:
            return True
    def create(self, request, *args, **kwargs):
        serializer = self.get_serializer(data=request.data)
        user_id = request.data.get('user_id')
        class_id = request.data.get('class_id')
        print(user_id, class_id)
        a = set(ClassUser.objects.filter(status=1, class_id=class_id, user_id = user_id))
        print(a)
        if len(a)>0:
            return Response({"message":"Học viên đã ở trong lớp"})
        serializer.is_valid(raise_exception=True)
        self.perform_create(serializer)
        usr = User.objects.get(id=user_id)
        usr.usrClass = Class.objects.get(id=class_id).classname
        usr.save()
        headers = self.get_success_headers(serializer.data)
        return Response(serializer.data, status=status.HTTP_201_CREATED, headers=headers)
    def update(self, request, *args, **kwargs):
        partial = kwargs.pop('partial', False)
        instance = self.get_object()
        serializer = self.get_serializer(instance, data=request.data, partial=partial)
        serializer.is_valid(raise_exception=True)
        user_id = request.data.get('user_id')
        class_id = request.data.get('class_id')
        usr = User.objects.get(id=user_id)
        usr.usrClass = Class.objects.get(id=class_id).classname
        usr.save()
        self.perform_update(serializer)

        if getattr(instance, '_prefetched_objects_cache', None):
            # If 'prefetch_related' has been applied to a queryset, we need to
            # forcibly invalidate the prefetch cache on the instance.
            instance._prefetched_objects_cache = {}

        return Response(serializer.data)
    def destroy(self, request, *args, **kwargs):
        instance = self.get_object()
        if (instance.status == True):
            obj =instance.user_id
            obj.chose_class = False
            obj.save()
        self.perform_destroy(instance)
        return Response(status=status.HTTP_204_NO_CONTENT)

    def perform_destroy(self, instance):
        instance.delete()
class DanhSachHocVienChuaCoLop(generics.ListAPIView):
    authentication_classes = (CsrfExemptSessionAuthentication,)
    pagination_class = LargeResultsSetPagination
    serializer_class = User2Serializer
    queryset = User.objects.all()
    def get_queryset(self):
        
        
        user_have = ClassUser.objects.filter(status=1).values_list('user_id', flat=True).distinct()
        ids  = User.objects.filter(roleid=3).values_list('id', flat=True).distinct()
        
        return User.objects.filter(id__in=list(set(ids) - set(user_have)))
class ApproveChangeClassView(generics.UpdateAPIView):
    authentication_classes = (CsrfExemptSessionAuthentication,)
class DanhSachLopGvPhuTrach(generics.ListAPIView):
    authentication_classes = (CsrfExemptSessionAuthentication,)
    serializer_class = ClassesSerializer
    # permission_classes = [IsTeacher, IsAdmin, IsStudent]
    pagination_class = LargeResultsSetPagination
    def get_queryset(self):
        user = self.request.query_params.get('user_id')
        print(user)
        class_users = ClassUser.objects.filter(user_id=user, status=1)
        class_ids = class_users.values_list('class_id', flat=True).distinct()
        return Class.objects.filter(classid__in=class_ids)
    user_id = openapi.Parameter(
    'user_id', openapi.IN_QUERY, description='id cua user', type=openapi.TYPE_NUMBER)

    @swagger_auto_schema(method='get',manual_parameters=[user_id],responses={404: 'Not found', 200:'ok'})
    @action(methods=['get'], detail=False)
    def get(self, request, *args, **kwargs):
        
        # If a class ID is provided in the URL, return the list of students in the class , user_id__roleid=3
        class_id = self.kwargs.get('pk')
        if class_id:
            class_users = ClassUser.objects.filter(class_id=class_id, status=1)
            print(class_users)
            students = [class_user.user_id for class_user in class_users]
            if None in students: students.remove(None)
            paginator = self.pagination_class()
            paginated_queryset = paginator.paginate_queryset(students, request)
            serializer = UserSerializer(paginated_queryset, many=True)
            
            return paginator.get_paginated_response(serializer.data)

        # Otherwise, return the list of classes for the authenticated teacher
        queryset = self.get_queryset()
        # Paginate the queryset
        paginator = self.pagination_class()
        paginated_queryset = paginator.paginate_queryset(queryset, request)
        
        serializer = ClassesSerializer(paginated_queryset, many=True)
        return paginator.get_paginated_response(serializer.data)

class ChangePasswordView(generics.UpdateAPIView):
    authentication_classes = (CsrfExemptSessionAuthentication,)
    """
    An endpoint for changing password.
    """
    serializer_class = ChangePasswordSerializer
    model = User
    permission_classes = (IsAuthenticated,)

    def get_object(self, queryset=None):
        obj = self.request.user
        return obj

    @swagger_auto_schema(tags=['Đăng nhập - Đăng ký'])
    def update(self, request, *args, **kwargs):
        self.object = self.get_object()
        serializer = self.get_serializer(data=request.data)

        if serializer.is_valid():
            # Check old password
            if not self.object.check_password(serializer.data.get("old_password")):
                return Response({"old_password": ["Wrong password."]}, status=status.HTTP_400_BAD_REQUEST)
            # set_password also hashes the password that the user will get
            self.object.set_password(serializer.data.get("new_password"))
            self.object.save()
            response = {
                'status': 'success',
                'code': status.HTTP_200_OK,
                'message': 'Password updated successfully',
                'data': []
            }

            return Response(response)

        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)


class ChangeNameView(generics.UpdateAPIView):
    """
    An endpoint for changing name.
    """
    serializer_class = ChangeNameSerializer
    model = User
    permission_classes = (IsAuthenticated,)
    authentication_classes = (CsrfExemptSessionAuthentication,)

    def get_object(self, queryset=None):
        obj = self.request.user
        obj = User.objects.get(id=obj.pk)
        return obj

    def update(self, request, *args, **kwargs):
        self.object = self.get_object()
        serializer = self.get_serializer(data=request.data)

        if serializer.is_valid():
            # Check password
            # if not self.object.check_password(serializer.data.get("password")):
            #     return Response({"password": ["Wrong password."]}, status=status.HTTP_400_BAD_REQUEST)
            # set_password also hashes the password that the user will get
            self.object.name = serializer.data.get("name")
            # self.object.usrclass.set(serializer.data.get("usrclass"))
            self.object.usrdob = serializer.data.get("usrdob")
            self.object.usrfullname = serializer.data.get("usrfullname")
            self.object.usrfaculty = serializer.data.get("usrfaculty")

            self.object.save()
            response = {
                'status': 'success',
                'code': status.HTTP_200_OK,
                'message': 'Infor updated successfully',
                'data': []
            }

            return Response(response)

        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)


class UserView(generics.RetrieveAPIView):
    serializer_class = UserSerializer
    lookup_field = 'pk'
    authentication_classes = (CsrfExemptSessionAuthentication,)
    def get_userclass_classname(self, user_id):
        classuser = ClassUser.objects.filter(user_id=user_id, status=1).first()
        if classuser:
            classname = classuser.class_id.classname
        else:
            classname = None
        return classname
    def get_object(self, *args, **kwargs):
        infor = User.objects.get(id=self.request.user.id)
        infor.usrclass = self.get_userclass_classname(infor.id)
        return infor
    # # permission_classes = [permissions.IsAuthenticatedOrReadOnly]
    # # permission_classes = [permissions.IsAuthenticatedOrReadOnly,
    # #                       IsOwnerOrReadOnly]
    # test_param1 = openapi.Parameter('check_type', openapi.IN_QUERY, description="height or width", type=openapi.TYPE_STRING)
    # test_param2 = openapi.Parameter('id', openapi.IN_QUERY, description="id of expAI", type=openapi.TYPE_NUMBER)
    # user_response = openapi.Response('response description', StudientSerializer)

    # @swagger_auto_schema(method='get', manual_parameters=[test_param1, test_param2], responses={404: 'Not found', 200:'ok', 201:StudientSerializer})
    # @action(methods=['GET'], detail=False, url_path='check-body')
    # def check_body(self, request):
    #     """
    #     Check body API
    #     """
    #     check_type = request.query_params.get('check_type')
    #     id = request.query_params.get('id')

    #     print(check_type, id)
    #     obj = self.queryset.get(id=id)
    #     rs = ""
    #     if check_type == 'height':
    #         if obj.height > 1:
    #             rs = "tall"
    #         else:
    #             rs = "short"
    #     else:
    #         if obj.weight > 1:
    #             rs = "fat"
    #         else:
    #             rs = "thin"
    #     return Response({"result": rs})

    # @action(methods=['POST'], detail=False)
    # def echo(self, request):
    #     # deserializer
    #     obj = StudientSerializer(request.data)
    #     print(obj)
    #     # serializer
    #     return Response(obj.data)

    # @action(methods=['GET'], detail=False)
    # def find_by_name(self, request):
    #     name = request.query_params.get('name')
    #     print(self.queryset)
    #     objs = expAI.objects.filter(name__exact=name)
    #     return Response(StudientSerializer(objs).data)


class ExperimentsViewSet(viewsets.ModelViewSet):
    model = Experiments
    queryset = Experiments.objects.all()
    serializer_class = ExperimentsSerializer
    authentication_classes = (CsrfExemptSessionAuthentication,)
    pagination_class = LargeResultsSetPagination
    permission_classes = [IsOwnerExp | IsAdmin]
    authentication_classes = (CsrfExemptSessionAuthentication,)

    def list(self, request, *args, **kwargs):
        '''
        List all items
        '''
        if request.user.id == None:
            return Response(status=status.HTTP_401_UNAUTHORIZED)
        usr = request.user
        usr = User.objects.get(id=usr.pk)
        a = ClassUser.objects.filter(status=1).filter(user_id=usr.id)
        b = list(a.values_list('class_id', flat=True))
        c = sum([list(ClassUser.objects.filter(status=1).filter(
            class_id=i).values_list("user_id", flat=True)) for i in b], [])
        d = User.objects.filter(is_active=1, id__in=c)
        if usr.roleid.rolename == "ADMIN":
            queryset = Experiments.objects.all()
        elif usr.roleid.rolename == "STUDENT":
            queryset = Experiments.objects.filter(expcreatorid=usr.id)
        else:  # giao vien
            usrclass = list(usr.usrclass.all())
            student = [list(i.user_set.all()) for i in usrclass]
            student = sum(student, [])
            queryset = Experiments.objects.filter(
                expcreatorid__in=d) | Experiments.objects.filter(expcreatorid=usr.id)
        page = self.paginate_queryset(queryset)
        if page is not None:
            serializer = self.get_serializer(page, many=True)
            return self.get_paginated_response(serializer.data)
        serializer = ExperimentsSerializer(queryset, many=True)
        return Response(serializer.data)

    def create(self, request, *args, **kwargs):

        if request.user.id == None:
            return Response(status=status.HTTP_401_UNAUTHORIZED)
        serializer = ExperimentsSerializer(data=request.data)
        if serializer.is_valid():
            myexp = serializer.save()
            myexp.expcreatorid = request.user
            myexp.expstatus = 1
            myexp.save()
            serializer = ExperimentsSerializer(myexp, many=False)
            return Response(serializer.data, status=status.HTTP_201_CREATED)

#                return JsonResponse({
#                    'message': 'Create a new exp successful!'
#                }, status=status.HTTP_201_CREATED)

        return JsonResponse({
            'message': 'Create a new exp unsuccessful!'
        }, status=status.HTTP_400_BAD_REQUEST)

    @swagger_auto_schema(method='get',manual_parameters=[],responses={404: 'Not found', 200:'ok'})
    @action(methods=['GET'], detail=False, url_path='get-sum-exps')
    def get_sum_exps(self, request):

        sum_exp = len(Experiments.objects.all())
        list_Softwarelibs = Softwarelibs.objects.all()
        serializer = SoftwareLibsSerializer(list_Softwarelibs, many=True)
        list_sum_by_sortlib = []
        for software in list_Softwarelibs:
            number_exp_of_software = len(Experiments.objects.filter(expsoftwarelibid = software))
            dict = {}
            dict['name'] = software.softwarelibname
            dict['id'] = software.softwarelibid
            dict['number'] = number_exp_of_software
            list_sum_by_sortlib.append(dict)
        


        response = {
            'status': 'success',
            'code': status.HTTP_200_OK,
            'message': 'Data uploaded successfully',
            'data': {'sum': sum_exp, 'list_sum_by_sortlib': list_sum_by_sortlib }
        }
        return Response(response, status=status.HTTP_200_OK)
    

    id_exp = openapi.Parameter(
    'id_exp', openapi.IN_QUERY, description='id cua exp', type=openapi.TYPE_NUMBER)

    id_softlib = openapi.Parameter(
    'id_softlib', openapi.IN_QUERY, description='id cua softlib', type=openapi.TYPE_NUMBER)
    keyword = openapi.Parameter(
    'keyword', openapi.IN_QUERY, description='keyword tìm kiếm', type=openapi.TYPE_STRING)

    @swagger_auto_schema(method='get', manual_parameters=[id_softlib,keyword], responses={404: 'Not found', 200: 'ok', 201: ExperimentsSerializer})
    @action(methods=['GET'], detail=False, url_path='search_exp')
    def search_exp(self, request):
        id_softlib = request.query_params.get('id_softlib')
        keyword = request.query_params.get('keyword')

        if request.user.id == None:
            return Response(status=status.HTTP_401_UNAUTHORIZED)
        usr = request.user
        usr = User.objects.get(id=usr.pk)
        a = ClassUser.objects.filter(status=1).filter(user_id=usr.id)
        b = list(a.values_list('class_id', flat=True))
        c = sum([list(ClassUser.objects.filter(status=1).filter(
            class_id=i).values_list("user_id", flat=True)) for i in b], [])
        d = User.objects.filter(is_active=1, id__in=c)
        if usr.roleid.rolename == "ADMIN":
            if id_softlib != None:
                queryset = Experiments.objects.filter(expsoftwarelibid = id_softlib,expname__contains = keyword)

            else:
                queryset = Experiments.objects.filter(expname__contains = keyword)
        elif usr.roleid.rolename == "STUDENT":

            if id_softlib != None:
                queryset = Experiments.objects.filter(expcreatorid=usr.id,expsoftwarelibid = id_softlib,expname__contains = keyword)
            else:
                queryset = Experiments.objects.filter(expcreatorid=usr.id,expname__contains = keyword)
        else:  # giao vien
            usrclass = list(usr.usrclass.all())
            student = [list(i.user_set.all()) for i in usrclass]
            student = sum(student, [])
            if id_softlib != None:
                queryset = Experiments.objects.filter(
                    expcreatorid__in=d,expsoftwarelibid = id_softlib,expname__contains = keyword) | Experiments.objects.filter(expcreatorid=usr.id,expsoftwarelibid = id_softlib,expname__contains = keyword)
            else:
                                queryset = Experiments.objects.filter(
                    expcreatorid__in=d,expname__contains = keyword) | Experiments.objects.filter(expcreatorid=usr.id,expname__contains = keyword)
        page = self.paginate_queryset(queryset)
        if page is not None:
            serializer = self.get_serializer(page, many=True)
            return self.get_paginated_response(serializer.data)
        serializer = ExperimentsSerializer(queryset, many=True)
        return Response(serializer.data)

    @swagger_auto_schema(method='get',manual_parameters=[id_exp],responses={404: 'Not found', 200:'ok'})
    @action(methods=['GET'], detail=False, url_path='get-ranking-by-exp')
    def get_ranking_by_exp(self, request):
        id_exp = request.query_params.get('id_exp')
        _exp = Experiments.objects.get(pk = id_exp)

        list_config = Paramsconfigs.objects.filter(configexpid = _exp)
        list_train_result = []
        list_test_result = []

        for config in list_config:
            result_train = Trainningresults.objects.filter(configid = config,is_last = True).last()
            result_test = Results.objects.filter(resultconfigid = config).last()
            if result_train != None and result_test != None :
                list_test_result.append(result_test.resultaccuracy)
                list_train_result.append(result_train.accuracy)
            elif result_train != None:
                list_test_result.append(result_train.accuracy)
                list_train_result.append(result_train.accuracy)
            elif result_test != None:
                list_test_result.append(result_test.resultaccuracy)
                list_train_result.append(result_test.resultaccuracy)
            else:
                continue


            

        return_list = zip(ParamsconfigsSerializer(list_config,many = True).data,list_train_result,list_test_result)





        


        response = {
            'status': 'success',
            'code': status.HTTP_200_OK,
            'message': 'Data uploaded successfully',
            'data': {'list_result':  return_list  }
        }
        return Response(response, status=status.HTTP_200_OK)


    @swagger_auto_schema(method='get',manual_parameters=[],responses={404: 'Not found', 200:'ok'})
    @action(methods=['GET'], detail=False, url_path='draw-exp-chart')
    def draw_exp_chart(self, request):
        _exps = Experiments.objects.all().order_by('expcreatedtime')
        start = _exps.first()
        end = _exps.last()
        return_list = []
        days = (end.expcreatedtime-start.expcreatedtime).days

        for i in range(0,days+2):
            cur_time  = start.expcreatedtime +datetime.timedelta(days=i)
            dict = {}
            dict['time'] = cur_time.date()
            dict['count_exp'] = Experiments.objects.filter(expcreatedtime__startswith = cur_time.date()).count()
            return_list.append(dict)

        response = {
            'status': 'success',
            'code': status.HTTP_200_OK,
            'message': 'Successfully',
            'data': {'list_result':  return_list  }
        }
        return Response(response, status=status.HTTP_200_OK)


    @swagger_auto_schema(method='get', manual_parameters=[id_softlib], responses={404: 'Not found', 200: 'ok', 201: ModelsSerializer})
    @action(methods=['GET'], detail=False, url_path='get-list-models')
    def get_list_models(self, request):
        """
        lay ds model theo id softlib
        """
        if request.user.id == None:
            return Response(status=status.HTTP_401_UNAUTHORIZED)

        id_softlib = request.query_params.get('id_softlib')

        models = Models.objects.filter(modelsoftlibid=id_softlib)
        serializer = ModelsSerializer(models, many=True)

        return Response(serializer.data)

    #id_softlib = openapi.Parameter('id_softlib',openapi.IN_QUERY,description='id cua softlib',type=openapi.TYPE_NUMBER)
    @swagger_auto_schema(method='get', manual_parameters=[id_softlib], responses={404: 'Not found', 200: 'ok', 201: DatasetsSerializer})
    @action(methods=['GET'], detail=False, url_path='get-list-dataset')
    def get_list_datasets(self, request):
        if request.user.id == None:
            return Response(status=status.HTTP_401_UNAUTHORIZED)
        usr = self.request.user
        try:
            usr = User.objects.get(email=usr.email)
        except:
            print("User not authorized")
            return
        a = ClassUser.objects.filter(status=1).filter(user_id=usr.id)
        b = list(a.values_list('class_id', flat=True))
        c = sum([list(ClassUser.objects.filter(status=1).filter(
            class_id=i).values_list("user_id", flat=True)) for i in b], [])
        d = User.objects.filter(is_active=1, id__in=c)
        id_softlib = request.query_params.get('id_softlib')

        if usr.roleid.rolename == "ADMIN":
            queryset = Datasets.objects.all()
        elif usr.roleid.rolename == "STUDENT":
            queryset = Datasets.objects.filter(datasettype=1, datasetsoftID__pk=id_softlib) | Datasets.objects.filter(
                datasetowner=self.request.user, datasetsoftID__pk=id_softlib)
        else:  # giao vien
            usrclass = list(usr.usrclass.all())
            student = [list(i.user_set.all()) for i in usrclass]
            student = sum(student, [])
            queryset = Datasets.objects.filter(datasettype=1, datasetsoftID__pk=id_softlib) | Datasets.objects.filter(
                datasetowner__in=d, datasetsoftID__pk=id_softlib)

        serializer = DatasetsSerializer(queryset, many=True)

        return Response(serializer.data)
    id_model = openapi.Parameter(
        'id_model', openapi.IN_QUERY, description='id model', type=openapi.TYPE_NUMBER)

    @swagger_auto_schema(method='get', manual_parameters=[id_model], responses={404: 'Not found', 200: 'ok', 201: ModelsSerializer})
    @action(methods=['GET'], detail=False, url_path='get-default-parameters')
    def get_default_parameters(self, request):
        """
        set-parameters
        """
        if request.user.id == None:
            return Response(status=status.HTTP_401_UNAUTHORIZED)

        id_model = request.query_params.get('id_model')

        models = Models.objects.get(modelid=id_model)
        serializer = ModelsSerializer(models, many=False)

        return Response(serializer.data)

    # start - stop train

    paramsconfigs_json = openapi.Parameter(
        'paramsconfigs_json', openapi.IN_QUERY, description='json string paramsconfig', type=openapi.TYPE_STRING)
    id_paramsconfigs = openapi.Parameter(
        'id_paramsconfigs', openapi.IN_QUERY, description='id cua bang paramsconfig', type=openapi.TYPE_NUMBER)

    @swagger_auto_schema(manual_parameters=[id_exp, paramsconfigs_json], responses={404: 'Not found', 200: 'ok'})
    @action(methods=['GET'], detail=False, url_path='start-train')
    def start_train(self, request):
        """
        start train
        """
        if request.user.id == None:
            return Response(status=status.HTTP_401_UNAUTHORIZED)

        # user = request.user
        # user = User.objects.get( id = user.pk)

        id_exp = request.query_params.get('id_exp')
        paramsconfigs_json = request.query_params.get('paramsconfigs_json')
        # print(paramsconfigs_json)

        if check_json_file(paramsconfigs_json):

            exp = Experiments.objects.get(expid=id_exp)
            exp.expstatus = 2
            paramsconfigs = Paramsconfigs(
                jsonstringparams=paramsconfigs_json, trainningstatus=1, configexpid=exp)
            paramsconfigs.save()
            exp.save()
            id_params = paramsconfigs.pk
            dataset = Datasets.objects.get(pk = exp.expdatasetid.pk)



            # --------------------
            # thread
            import threading
            _model = Models.objects.get(pk = exp.expmodelid.pk)
            print("------------------------------")
            print(_model.modelname)
            if int(_model.modelid) == 1:
                t = threading.Thread(target=Retina_mnet.train,
                                    args=(id_params,dataset.datasetfolderurl,paramsconfigs_json), kwargs={})
            elif int(_model.modelid) == 2:
                t = threading.Thread(target=Retina_mnet.train,
                    args=(id_params,dataset.datasetfolderurl,paramsconfigs_json), kwargs={})
            elif int(_model.modelid) == 3:
                t = threading.Thread(target=FaceNet.train,
                    args=(id_params,dataset.datasetfolderurl,paramsconfigs_json), kwargs={})
            
            elif int(_model.modelid) == 4:
                t = threading.Thread(target=fightdetection.train,
                    args=(id_params,dataset.datasetfolderurl,paramsconfigs_json), kwargs={})
            else:
                return JsonResponse({
                'message': 'Mô hình không hợp lệ! '
                 }, status=status.HTTP_400_BAD_REQUEST)
                
            t.setDaemon(True)

            #-----fake kq
            paramsconfigs.trainningstatus = 0
            _new_result = Trainningresults()
            _new_result.configid = paramsconfigs
            _new_result.accuracy = 0.7
            _new_result.lossvalue = 1.5
            _new_result.trainresultindex = 1
            _new_result.is_last = False
            _new_result.save()
            _new_result = Trainningresults()
            _new_result.configid = paramsconfigs
            _new_result.accuracy = 0.8
            _new_result.lossvalue = 1
            _new_result.trainresultindex = 2
            _new_result.is_last = False
            _new_result.save()
            _new_result = Trainningresults()
            _new_result.configid = paramsconfigs
            _new_result.accuracy = 0.9
            _new_result.lossvalue = 0.75
            _new_result.trainresultindex = 3
            _new_result.is_last = True
            _new_result.save()
            paramsconfigs.save()
            #-------------------------


            #t.start()
            # --------------------

            serializer = ParamsconfigsSerializer(paramsconfigs, many=False)

            return Response(serializer.data, status=status.HTTP_201_CREATED)
        else:
            return JsonResponse({
                'message': 'Có một số lỗi với chuỗi json được nhập!'
            }, status=status.HTTP_400_BAD_REQUEST)
            # return Response(status=status.HTTP_400_BAD_REQUEST)

    @swagger_auto_schema(manual_parameters=[id_exp], responses={404: 'Not found', 200: 'ok'})
    @action(methods=['GET'], detail=False, url_path='list-paramsconfigs')
    def list_paramsconfigs(self, request):
        if request.user.id == None:
            return Response(status=status.HTTP_401_UNAUTHORIZED)

        id_exp = request.query_params.get('id_exp')

        paramsconfigs = Paramsconfigs.objects.filter(configexpid=id_exp)

        serializer = ParamsconfigsSerializer(paramsconfigs, many=True)

        return Response(serializer.data, status=status.HTTP_200_OK)

    @swagger_auto_schema(manual_parameters=[id_paramsconfigs], responses={404: 'Not found', 200: 'ok'})
    @action(methods=['GET'], detail=False, url_path='stop-train')
    def stop_train(self, request):
        """
        stop train
        """
        if request.user.id == None:
            return Response(status=status.HTTP_401_UNAUTHORIZED)

        user = request.user
        user = User.objects.get(id=user.pk)

        # id_exp = request.query_params.get('id_exp')
        id_paramsconfigs = request.query_params.get('id_paramsconfigs')

        # exp = Experiments.objects.get(expid=id_exp)
        # exp.expstatus = 2
        paramsconfigs = Paramsconfigs.objects.get(configid=id_paramsconfigs)
        paramsconfigs.trainningstatus = 0
        paramsconfigs.save()
        exp = paramsconfigs.configexpid
        exp.expstatus = 2
        exp.save()
        _results = Trainningresults.objects.filter(
            configid=paramsconfigs).order_by('trainresultid').last()
        serializer = TrainningresultsSerializer(_results, many=False)

        return Response(serializer.data, status=status.HTTP_200_OK)

    @swagger_auto_schema(method='get', manual_parameters=[id_paramsconfigs], responses={404: 'Not found', 200: 'ok', 201: ResultsSerializer})
    @action(methods=['GET'], detail=False, url_path='get-all-traning-results')
    def get_all_traning_results(self, request):
        """
        get trainning results
        """
        if request.user.id == None:
            return Response(status=status.HTTP_401_UNAUTHORIZED)

        # user = request.user
        # user = User.objects.get( id = user.pk)

        # id_exp = request.query_params.get('id_exp')
        id_paramsconfigs = request.query_params.get('id_paramsconfigs')

        # exp = Experiments.objects.filter(expid = id_exp)
        paramsconfigs = Paramsconfigs.objects.get(configid=id_paramsconfigs)
        queryset = Trainningresults.objects.filter(
            configid=paramsconfigs).order_by('trainresultid')

        serializer = TrainningresultsSerializer(queryset, many=True)
        return Response(serializer.data)

    pre_result_index = openapi.Parameter('pre_result_index', openapi.IN_QUERY,
                                         description='index của bản ghi trước đó, nếu gọi lần đầu thì để là 0', type=openapi.TYPE_NUMBER)

    @swagger_auto_schema(method='get', manual_parameters=[id_paramsconfigs, pre_result_index], responses={404: 'Not found', 200: 'ok', 201: ResultsSerializer})
    @action(methods=['GET'], detail=False, url_path='get-new-traning-result')
    def get_new_traning_result(self, request):
        """
        get new trainning results
        """
        if request.user.id == None:
            return Response(status=status.HTTP_401_UNAUTHORIZED)

        # user = request.user
        # user = User.objects.get( id = user.pk)

        # id_exp = request.query_params.get('id_exp')
        id_paramsconfigs = request.query_params.get('id_paramsconfigs')
        pre_result_index = request.query_params.get('pre_result_index')

        # exp = Experiments.objects.filter(expid = id_exp)
        paramsconfigs = Paramsconfigs.objects.get(configid=id_paramsconfigs)
        _results = Trainningresults.objects.filter(
            configid=paramsconfigs, trainresultindex__gt=pre_result_index).order_by('trainresultindex')

        if _results:
            queryset = _results
            serializer = TrainningresultsSerializer(queryset, many=True)
            response = {
                'status': 'success',
                'code': status.HTTP_201_CREATED,
                'message': 'Data uploaded successfully',
                'data': {'result': serializer.data, 'status': paramsconfigs.trainningstatus}
            }
            return Response(response, status=status.HTTP_200_OK)
        else:
            return JsonResponse({
                'message': 'Chưa có result mới!'
            }, status=status.HTTP_102_PROCESSING)

    id_dataset = openapi.Parameter(
        'id_dataset', openapi.IN_QUERY, description='id cua dataset test', type=openapi.TYPE_NUMBER)
    id_paramsconfigs = openapi.Parameter(
        'id_paramsconfigs', openapi.IN_QUERY, description='id cua bang paramsconfig', type=openapi.TYPE_NUMBER)

    @swagger_auto_schema(manual_parameters=[id_dataset, id_paramsconfigs], responses={404: 'Not found', 200: 'ok', 201: ResultsSerializer})
    @action(methods=['GET'], detail=False, url_path='start-test')
    def start_test(self, request):
        """
        start_test
        """
        if request.user.id == None:
            return Response(status=status.HTTP_401_UNAUTHORIZED)

        id_dataset = request.query_params.get('id_dataset')
        id_paramsconfigs = request.query_params.get('id_paramsconfigs')
        _dataset = Datasets.objects.get(pk=id_dataset)
        _paramsconfigs = Paramsconfigs.objects.get(pk=id_paramsconfigs)
        _exp = _paramsconfigs.configexpid
        _exp.expstatus = 3
        _exp.save()
        _result = Results()
        _result.resultconfigid = _paramsconfigs
        _result.resulttestingdataset = _dataset
        #--fake kq----------------------
        _result.resultaccuracy = 0.85
        _result.resultdetail = " easy: 0.92; medium: 0.84; hard:0.71 "
        _result.save()
        #-----fake kq-------------------------

        _result.save()
        resultId = int(_result.pk)

        #----------------------------------- thread
        import threading
        _model = Models.objects.get(pk = _exp.expmodelid.pk)
        if int(_model.modelid) == 1:
            t = threading.Thread(target=Retina_mnet.test,
                args=(resultId,_dataset.datasetfolderurl), kwargs={})
        elif int(_model.modelid) == 2:
            t = threading.Thread(target=Retina_mnet.test,
                args=(resultId,_dataset.datasetfolderurl), kwargs={})
        elif int(_model.modelid) == 3:
            t = threading.Thread(target=FaceNet.test,
                args=(resultId,_dataset.datasetfolderurl), kwargs={})
        elif int(_model.modelid) == 4:
            t = threading.Thread(target=fightdetection.test,
                args=(resultId,_dataset.datasetfolderurl), kwargs={})
        else:
            return JsonResponse({
                'message': 'ID model k hợp lệ !'
            }, status=status.HTTP_400_BAD_REQUEST)
        t.setDaemon(True)
        #t.start()
        # -------------------------------------------------------------------
        serializer = ResultsSerializer(_result, many=False)
        return Response(serializer.data, status=status.HTTP_201_CREATED)

    id_test_result = openapi.Parameter('id_test_result', openapi.IN_QUERY,
                                       description='id test result nhan khi gọi API start test', type=openapi.TYPE_NUMBER)

    @swagger_auto_schema(manual_parameters=[id_test_result, ], responses={404: 'Not found', 200: 'ok', 201: ResultsSerializer})
    @action(methods=['GET'], detail=False, url_path='get-test-result')
    def get_test_result(self, request):
        if request.user.id == None:
            return Response(status=status.HTTP_401_UNAUTHORIZED)
        id_test_result = request.query_params.get('id_test_result')
        _result = Results.objects.get(pk=id_test_result)
        if _result.resultaccuracy:
            serializer = ResultsSerializer(_result, many=False)
            return Response(serializer.data, status=status.HTTP_200_OK)
        else:
            serializer = ResultsSerializer(_result, many=False)
            return Response(serializer.data, status=status.HTTP_200_OK)

    input_path1 = openapi.Parameter('input_path1', openapi.IN_QUERY,
                                    description='đường dẫn tới folder input', type=openapi.TYPE_STRING)
    input_path2 = openapi.Parameter('input_path2', openapi.IN_QUERY,
                                    description='đường dẫn tới folder ảnh người trong face Recognition', type=openapi.TYPE_STRING)
    data_type = openapi.Parameter(
        'data_type', openapi.IN_QUERY, description='image/video', type=openapi.TYPE_STRING)

    @swagger_auto_schema(manual_parameters=[id_paramsconfigs, input_path1, input_path2, data_type], responses={404: 'Not found', 200: 'ok', 201: PredictSerializer})
    @action(methods=['GET'], detail=False, url_path='predict')
    def predict(self, request):
        if request.user.id == None:
            return Response(status=status.HTTP_401_UNAUTHORIZED)
        input_path1 = request.query_params.get('input_path1')
        input_path2 = request.query_params.get('input_path2')
        data_type = request.query_params.get('data_type')
        id_paramsconfigs = request.query_params.get('id_paramsconfigs')

        _para = Paramsconfigs.objects.get(pk=id_paramsconfigs)

        _exp = Experiments.objects.get(pk = _para.configexpid.pk)
        _exp.expstatus = 4
        _exp.save()
        _predict = Predict()
        _predict.configid = _para
        _predict.inputpath = input_path1
        if input_path2:
            _predict.inputpath2 = input_path2
        _predict.datatype = data_type
        _predict.save()

        import threading
        trained_model = str(_para.configaftertrainmodelpath)
        #--fake kq -------------------------------
        trained_model = "./expAI/AI_models/Pytorch_Retinaface/weights/mobilenet0.25_Final.pth"

        #------------------------------------------
        _model = Models.objects.get(pk = _exp.expmodelid.pk)
        if int(_model.modelid) == 1:
            t = threading.Thread(target=Retina_mnet.predict,
                                args=(_predict.pk,trained_model), kwargs={})
        elif int(_model.modelid) == 2:
            t = threading.Thread(target=Retina_mnet.predict,
                args=(_predict.pk,trained_model), kwargs={})
        elif int(_model.modelid) == 3:
            t = threading.Thread(target=FaceNet.predict,
                args=(_predict.pk,trained_model), kwargs={})
        elif int(_model.modelid) == 4:
            t = threading.Thread(target=fightdetection.predict,
                args=(_predict.pk,trained_model), kwargs={})
        else:
            return JsonResponse({
                'message': 'Có một số lỗi với chuỗi json được nhập!'
            }, status=status.HTTP_400_BAD_REQUEST)
        t.setDaemon(True)
        t.start()

        serializer = PredictSerializer(_predict, many=False)
        return Response(serializer.data, status=status.HTTP_200_OK)

    id_predict = openapi.Parameter(
        'id_predict', openapi.IN_QUERY, description='id predict', type=openapi.TYPE_NUMBER)

    @swagger_auto_schema(manual_parameters=[id_predict], responses={404: 'Not found', 200: 'ok', 201: PredictSerializer})
    @action(methods=['GET'], detail=False, url_path='get_predict_result')
    def get_predict_result(self, request):
        if request.user.id == None:
            return Response(status=status.HTTP_401_UNAUTHORIZED)

        id_predict = request.query_params.get('id_predict')

        _predict = Predict.objects.get(pk=id_predict)
        list_result = []
        if _predict.outputpath == None:
            list_result = []
        elif os.path.exists(_predict.outputpath):
            list_result = os.listdir(_predict.outputpath)

        serializer = PredictSerializer(_predict, many=False)
        if len(list_result) > 0:
            response = {
                'status': 'success',
                'code': status.HTTP_201_CREATED,
                'message': 'Predict successfully',
                'data': {'result': list_result, 'data': serializer.data}
            }
        else:
            response = {
                'status': 'success',
                'code': status.HTTP_201_CREATED,
                'message': 'Output is Null',
                'data': {'result': list_result, 'data': serializer.data}
            }

        return Response(response, status=status.HTTP_200_OK)

    @swagger_auto_schema(manual_parameters=[id_exp], responses={404: 'Not found', 200: 'ok', 201: ResultsSerializer})
    @action(methods=['GET'], detail=False, url_path='get_list_test_results')
    def get_list_test_results(self, request):
        if request.user.id == None:
            return Response(status=status.HTTP_401_UNAUTHORIZED)

        id_exp = request.query_params.get('id_exp')
        _paramsconfigs = Paramsconfigs.objects.filter(configexpid=id_exp)
        _result = Results.objects.filter(resultconfigid__in=_paramsconfigs)

        list_serializer = ResultsSerializer(_result, many=True)
        for item in list_serializer.data:
            item['resultconfig'] = ParamsconfigsSerializer(
                Paramsconfigs.objects.get(pk=item['resultconfigid']), many=False).data
            item['resultdataset'] = DatasetsSerializer(Datasets.objects.get(
                pk=item['resulttestingdataset']), many=False).data

        return Response(list_serializer.data, status=status.HTTP_200_OK)

from django.http import FileResponse
from django.views import View

class DownloadView(views.APIView):
    configID = openapi.Parameter(
        'configID', openapi.IN_QUERY, description='id cua bang paramsconfig', type=openapi.TYPE_NUMBER)
    @swagger_auto_schema(
        operation_id='Download docx',
        operation_description='Download docx',
        operation_summary="Download docx",
        manual_parameters=[configID],
        tags=['experiment']
    )
    def get(self, request, *args, **kwargs):
        # Replace 'path/to/file' with the actual path to the file you want to serve.
        
        configID = request.query_params.get('configID')
        config = Paramsconfigs.objects.values('configexpid__expname',
                                              'configexpid__expcreatedtime',
                                              'jsonstringparams',
                                              'configexpid__expid',
                                              'configexpid__expcreatorid__name',
                                              'configexpid__expdatasetid__datasetname',
                                              'configexpid__expmodelid').get(configid=configID)
        trainresult = Trainningresults.objects.filter(configid=configID).values('trainresultindex', 'lossvalue', 'accuracy')
        testingresult = Results.objects.values('resultaccuracy', 'resulttestingdataset__datasetname').get(resultconfigid=configID)
        # print(trainresult)
        # Open the file and create a response object with its contents.
        file = gen_report_docx(expName=config['configexpid__expname'], expID=config['configexpid__expid'], expCreator=config['configexpid__expcreatorid__name'],
                    expCreatedTime=config['configexpid__expcreatedtime'], 
                    dataset=config['configexpid__expdatasetid__datasetname'],
                      test_dataset_name=testingresult['resulttestingdataset__datasetname'], 
                      test_dataset_acc=testingresult['resultaccuracy'], trainresult = trainresult)
        response = FileResponse(file)

        # Set the content type and content disposition headers for the response.
        response['Content-Type'] = 'application/octet-stream'
        response['Content-Disposition'] = 'attachment; filename="report.docx"'

        return response

class DatasetsUploadView(views.APIView):
    parser_classes = [FormParser, MultiPartParser]
    authentication_classes = (CsrfExemptSessionAuthentication,)
    # @swagger_auto_schema(tags=['datasets'])

    @swagger_auto_schema(
        operation_id='Upload zip',
        operation_description='Upload zip',
        operation_summary="Upload file zip cho bạn Hiếu",
        manual_parameters=[
            openapi.Parameter(
                'file', openapi.IN_FORM, type=openapi.TYPE_FILE, description='Zip to be uploaded'),

        ],
        tags=['datasets']
    )
    def post(self, request):
        file_obj = request.data['file']
        new_name = uuid.uuid4()

        with zipfile.ZipFile(file_obj, mode='r', allowZip64=True) as file:

            directory_to_extract = f"datasets/{new_name}"
            file.extractall(directory_to_extract)

        response = {
            'status': 'success',
            'code': status.HTTP_201_CREATED,
            'message': 'Data uploaded successfully',
            'data': new_name
        }
        return Response(response)


class ModelsViewSet(viewsets.ModelViewSet):
    queryset = Models.objects.all()
    serializer_class = ModelsSerializer
    # permission_classes = [IsOwner | IsAdmin]
    authentication_classes = (CsrfExemptSessionAuthentication,)
    def create(self, request, *args, **kwargs):
        usr = self.request.user
        if (usr.roleid.rolename == "STUDENT") and (request.data["modeltype"]==2):
            return Response("Học viên không được tạo public")
        serializer = self.get_serializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        self.perform_create(serializer)
        headers = self.get_success_headers(serializer.data)
        return Response(serializer.data, status=status.HTTP_201_CREATED, headers=headers)

    def perform_create(self, serializer):
        serializer.save(modelowner=self.request.user)

    id_user = openapi.Parameter('id_user', openapi.IN_QUERY,
                                description='id cua giao vien', type=openapi.TYPE_NUMBER)

    @swagger_auto_schema(method='get', manual_parameters=[id_user], responses={404: 'Not found', 200: 'ok', 201: ModelsSerializer})
    @action(methods=['GET'], detail=False, url_path='get-list-models')
    def get_list_models(self, request):
        """
        lay ds model theo id giao vien
        """
        if request.user.id == None:
            return Response(status=status.HTTP_401_UNAUTHORIZED)

        id_user = request.query_params.get('id_user')

        models = Models.objects.filter(modelcreator=id_user)
        serializer = ModelsSerializer(models, many=True)

        return Response(serializer.data)

    # def get_permissions(self):
    #     if self.action == 'get_list_model' or self.action == 'list':
    #         permission_classes = [IsStudent | IsTeacher | IsAdmin]
    #     else:
    #         permission_classes = [IsTeacher]

    #     return [permission() for permission in permission_classes]


class ModelsUploadView(views.APIView):
    parser_classes = [FormParser, MultiPartParser]
    permission_classes = [IsTeacher]
    authentication_classes = (CsrfExemptSessionAuthentication,)

    def post(self, request):
        file_obj = request.data['file']
        new_name = uuid.uuid4()

        with zipfile.ZipFile(file_obj, mode='r', allowZip64=True) as file:

            directory_to_extract = f"models/{new_name}"
            file.extractall(directory_to_extract)

        response = {
            'status': 'success',
            'code': status.HTTP_201_CREATED,
            'message': 'Data uploaded successfully',
            'data': new_name
        }
        return Response(response)

class Model_trainedViewSet(viewsets.ModelViewSet):
    queryset = Model_trained.objects.all()
    serializer_class = Model_trainedSerializer
    #permission_classes = [IsStudent | IsTeacher]
    authentication_classes = (CsrfExemptSessionAuthentication,)
    
    def get_serializer_class(self):
        if self.action == "create":
            return Creat_Model_trainedSerializer
        return Model_trainedSerializer
    
class FileUploadView(views.APIView):

    parser_classes = [FormParser, MultiPartParser]
    authentication_classes = (CsrfExemptSessionAuthentication,)

    @swagger_auto_schema(
        operation_id='Upload file',
        operation_description='Upload file',
        operation_summary="Upload a file",
        manual_parameters=[openapi.Parameter('file', openapi.IN_FORM, type=openapi.TYPE_FILE, description='file to be uploaded'), ], tags=['experiment']
    )
    def post(self, request):
        file_obj = request.FILES['file']
        new_name = uuid.uuid4()
        path = f"./media/predict_data/{new_name}/"
        if not os.path.exists(path):
            os.makedirs(path)
        with open(path + file_obj.name, 'wb+') as destination:
            for chunk in file_obj.chunks():
                destination.write(chunk)
        response = {
            'status': 'success',
            'code': status.HTTP_201_CREATED,
            'message': 'Data uploaded successfully',
            'data': path
        }
        return Response(response)


class FilesUploadView(views.APIView):

    parser_classes = [FormParser, MultiPartParser]
    authentication_classes = (CsrfExemptSessionAuthentication,)

    @swagger_auto_schema(
        operation_id='Upload files',
        operation_description='Upload files',
        operation_summary="Upload file files",
        manual_parameters=[openapi.Parameter('file', openapi.IN_FORM, type=openapi.TYPE_FILE, description='files to be uploaded'), ], tags=['experiment']
    )
    def post(self, request):
        new_name = uuid.uuid4()
        path = f"./media/predict_data/{new_name}/"
        if not os.path.exists(path):
            os.makedirs(path)

        for file_obj in request.FILES.getlist("files"):

            with open(path + file_obj.name, 'wb+') as destination:
                for chunk in file_obj.chunks():
                    destination.write(chunk)
        response = {
            'status': 'success',
            'code': status.HTTP_201_CREATED,
            'message': 'Data uploaded successfully',
            'data': path
        }
        return Response(response)
