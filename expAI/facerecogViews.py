from rest_framework import viewsets, status
from .models import *
from .serializers import FaceSerializer
from .permissions import *
from .paginations import *
from rest_framework import filters
from .filters import *
from rest_framework import views, generics, response, permissions, authentication
from django.http import JsonResponse
from rest_framework.decorators import action
from drf_yasg.utils import swagger_auto_schema
from drf_yasg import openapi
from rest_framework.parsers import FileUploadParser, FormParser, MultiPartParser
import os
import datetime
import uuid
from arcface import ArcFace
import numpy as np

class CsrfExemptSessionAuthentication(authentication.SessionAuthentication):
    def enforce_csrf(self, request):
        return

class FaceViewSet(viewsets.ModelViewSet):
    model = Face
    queryset = Face.objects.all()
    serializer_class = FaceSerializer
    authentication_classes = (CsrfExemptSessionAuthentication,)
    pagination_class = LargeResultsSetPagination
    permission_classes = [IsOwnerExp | IsAdmin]
    authentication_classes = (CsrfExemptSessionAuthentication,)

    def list(self, request, *args, **kwargs):
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
            queryset = Face.objects.all()
        elif usr.roleid.rolename == "STUDENT":
            queryset = Face.objects.filter(creator=usr.id)
        else:  # giao vien
            usrclass = list(usr.usrclass.all())
            student = [list(i.user_set.all()) for i in usrclass]
            student = sum(student, [])
            queryset = Face.objects.filter(
                creator__in=d) | Face.objects.filter(creator=usr.id)
        page = self.paginate_queryset(queryset)
        if page is not None:
            serializer = self.get_serializer(page, many=True)
            return self.get_paginated_response(serializer.data)
        serializer = FaceSerializer(queryset, many=True)
        return Response(serializer.data)
    

    # id_exp = openapi.Parameter(
    # 'id_exp', openapi.IN_QUERY, description='id cua exp', type=openapi.TYPE_NUMBER)

    # id_softlib = openapi.Parameter(
    # 'id_softlib', openapi.IN_QUERY, description='id cua softlib', type=openapi.TYPE_NUMBER)
    # keyword = openapi.Parameter(
    # 'keyword', openapi.IN_QUERY, description='keyword tìm kiếm', type=openapi.TYPE_STRING)

    

    # @swagger_auto_schema(method='get', manual_parameters=[id_softlib,keyword], responses={404: 'Not found', 200: 'ok', 201: FaceSerializer})
    # @action(methods=['GET'], detail=False, url_path='search_exp')

    def create(self, request, *args, **kwargs):

        if request.user.id == None:
            return Response(status=status.HTTP_401_UNAUTHORIZED)
        serializer = FaceSerializer(data=request.data)
        if serializer.is_valid():
            new_face = serializer.save()
            new_face.expcreatorid = request.user
            new_face.save()
            serializer = FaceSerializer(new_face, many=False)
            return Response(serializer.data, status=status.HTTP_201_CREATED)
        return JsonResponse({
            'message': 'Create a new face unsuccessful!'
        }, status=status.HTTP_400_BAD_REQUEST)

class FaceUploadView(views.APIView):

    parser_classes = [FormParser, MultiPartParser]
    authentication_classes = (CsrfExemptSessionAuthentication,)
    name = openapi.Parameter('name', openapi.IN_QUERY, description='tên đối tượng', type=openapi.TYPE_STRING)

    @swagger_auto_schema(
        operation_id='Upload face',
        operation_description='Upload face',
        operation_summary="Upload a face",
        manual_parameters=[openapi.Parameter('file', openapi.IN_FORM, type=openapi.TYPE_FILE, description='file to be uploaded'), name], tags=['facerecog']
    )
    def post(self, request):
        if request.user.id == None:
            return Response(status=status.HTTP_401_UNAUTHORIZED)
        file_obj = request.FILES['file']
        person_name = request.query_params.get('name')
        new_name = uuid.uuid4()
        path = f"./static/faces/{new_name}/"
        if not os.path.exists(path):
            os.makedirs(path)
        with open(path + file_obj.name, 'wb+') as destination:
            for chunk in file_obj.chunks():
                destination.write(chunk)
        faces = get_faces(path+file_obj.name)
        l_face = len(faces)
        if  l_face == 1:
            box = list(map(int, faces[0][:4]))
            face_rec = ArcFace.ArcFace()
            emb1 = face_rec.calc_emb(cv2.imread(path+file_obj.name)[box])
            bf = emb1.tobytes()
            new_face = Face()
            new_face.name = person_name
            new_face.image_path = path+file_obj.name
            new_face.x1 = box[0]
            new_face.y1 = box[1]
            new_face.x2 = box[2]
            new_face.y1 = box[3]
            new_face.creatorID = User.objects.get(pk = request.user.id)
            new_face.emb = bf
            new_face.save()
            response = {
            'status': 'success',
            'code': status.HTTP_201_CREATED,
            'message': 'image uploaded successfully',
            'data': FaceSerializer(new_face,many=False).data
        }
        elif l_face == 0 : 
            # The uploaded image doesn't contain a face, so we reject it
            os.remove(path+file_obj.name)
            response = {
            'status': 'success',
            'code': status.HTTP_200_OK,
            'message': 'no face in image'
        }
        else:
            os.remove(path+file_obj.name)
            response = {
            'status': 'success',
            'code': status.HTTP_200_OK,
            'message': '> 1 face in image'
        }

        return Response(response)
    
import cv2

def get_faces(image_path):
    print('image_path',image_path)
    detector = cv2.FaceDetectorYN.create("./weights/face_detection_yunet_2022mar.onnx", "", (320, 320))
    # Read image
    img = cv2.imread(image_path)
    # Get image shape
    img_W = int(img.shape[1])
    img_H = int(img.shape[0])
    # Set input size
    detector.setInputSize((img_W, img_H))
    # Getting detections
    detections = detector.detect(img)
    print(detections)
    faces = detections[1]
    try:
        print(len(faces))
    except:
        faces = []
    return faces