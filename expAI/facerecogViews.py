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
import requests
from bs4 import BeautifulSoup
import urllib.request
import json
from retinaface import RetinaFace

class CsrfExemptSessionAuthentication(authentication.SessionAuthentication):
    def enforce_csrf(self, request):
        return

class FaceViewSet(viewsets.ModelViewSet):
    model = Face
    queryset = Face.objects.all()
    serializer_class = FaceSerializer
    authentication_classes = (CsrfExemptSessionAuthentication,)
    pagination_class = LargeResultsSetPagination
    authentication_classes = (CsrfExemptSessionAuthentication,)
    def list(self, request, *args, **kwargs):
        if request.user.id == None:
            return Response(status=status.HTTP_401_UNAUTHORIZED)
        usr = request.user
        usr = User.objects.get(id=usr.pk)
        # a = ClassUser.objects.filter(status=1).filter(user_id=usr.id)
        # b = list(a.values_list('class_id', flat=True))
        # c = sum([list(ClassUser.objects.filter(status=1).filter(
        #     class_id=i).values_list("user_id", flat=True)) for i in b], [])
        # d = User.objects.filter(is_active=1, id__in=c)
        # if usr.roleid.rolename == "ADMIN":
        #     queryset = Face.objects.all()
        # elif usr.roleid.rolename == "STUDENT":
        #     queryset = Face.objects.filter(creatorID=usr.id)
        # else:  # giao vien
        #     usrclass = list(usr.usrclass.all())
        #     student = [list(i.user_set.all()) for i in usrclass]
        #     student = sum(student, [])
        #     queryset = Face.objects.filter(
        #         creatorID__in=d) | Face.objects.filter(creatorID=usr.id)

        queryset = Face.objects.filter(creatorID=usr)
        page = self.paginate_queryset(queryset)
        if page is not None:
            serializer = self.get_serializer(page, many=True)
            return self.get_paginated_response(serializer.data)
        serializer = FaceSerializer(queryset, many=True)
        return Response(serializer.data)
    
    web_url = openapi.Parameter('web_url', openapi.IN_QUERY, description='web URL', type=openapi.TYPE_STRING)
    score = openapi.Parameter('score', openapi.IN_QUERY, description='score', type=openapi.TYPE_NUMBER)
    @swagger_auto_schema(method='get', manual_parameters=[web_url,score], responses={404: 'Not found', 200: 'ok', 201: FaceSerializer})
    @action(methods=['GET'], detail=False, url_path='search_by_face')
    def search_face(self, request):
        web_url = request.query_params.get('web_url')
        score = request.query_params.get('score')
        score = float(score)
        if request.user.id == None:
            return Response(status=status.HTTP_401_UNAUTHORIZED)
        usr = request.user
        usr = User.objects.get(id=usr.pk)

        # a = ClassUser.objects.filter(status=1).filter(user_id=usr.id)
        # b = list(a.values_list('class_id', flat=True))
        # c = sum([list(ClassUser.objects.filter(status=1).filter(
        #     class_id=i).values_list("user_id", flat=True)) for i in b], [])
        # d = User.objects.filter(is_active=1, id__in=c)
        # if usr.roleid.rolename == "ADMIN":
        #     face_for_search = Face.objects.all()
        # elif usr.roleid.rolename == "STUDENT":
        #     face_for_search = Face.objects.filter(creatorID=usr.id)
        # else:  # giao vien
        #     usrclass = list(usr.usrclass.all())
        #     student = [list(i.user_set.all()) for i in usrclass]
        #     student = sum(student, [])
        #     face_for_search = Face.objects.filter(
        #         creatorID__in=d) | Face.objects.filter(creatorID=usr.id)
        face_for_search = Face.objects.filter(creatorID=usr)
        image_urls,image_crawleds = crawl_Web_URL(url=web_url)
        return_data = []
        face_rec = ArcFace.ArcFace()
        for url,img in zip(image_urls,image_crawleds):
            dict = {}
            dict['url'] = url
            print(url)
            try:
                faces_founds = get_faces_by_img(img)
            except:
                faces_founds = []
            dict['number_face'] = len(faces_founds)
            print(dict['number_face'])
            arr_face_found = []
            for face_found in faces_founds:
                face_dict = {}
                x1= face_found['x1']
                y1= face_found['y1']
                x2= face_found['x2']
                y2= face_found['y2']
                if x1 < 0:
                    x1 = 0
                if y1 < 0:
                    y1 =0
                # cv2.imwrite('test.jpg',img[box[1]:box[1]+box[3],box[0]:box[0]+box[2]])
                face_dict['box'] = [x1,y1,x2,y2]
                emb_f = face_rec.calc_emb(img[y1:y2,x1:x2])
                for face_point in face_for_search:
                    print('-------------------------------------------------------')
                    distance = cosine_distance(emb_f, np.frombuffer(face_point.emb, dtype=np.float32))
                    print(distance)
                    if distance > score:
                        face_dict['name'] = face_point.name
                        face_dict['score'] = distance
                        break
                    else:
                        face_dict['name'] = 'unknow'
                arr_face_found.append(face_dict)
            dict['arr_face'] = arr_face_found
            return_data.append(dict)


        return Response({
            'status': 'success',
            'code': status.HTTP_200_OK,
            'message': 'image uploaded successfully',
            'data': return_data
        })
    # id_exp = openapi.Parameter(
    # 'id_exp', openapi.IN_QUERY, description='id cua exp', type=openapi.TYPE_NUMBER)

    # id_softlib = openapi.Parameter(
    # 'id_softlib', openapi.IN_QUERY, description='id cua softlib', type=openapi.TYPE_NUMBER)
    # keyword = openapi.Parameter(
    # 'keyword', openapi.IN_QUERY, description='keyword tìm kiếm', type=openapi.TYPE_STRING)

    

    # @swagger_auto_schema(method='get', manual_parameters=[id_softlib,keyword], responses={404: 'Not found', 200: 'ok', 201: FaceSerializer})
    # @action(methods=['GET'], detail=False, url_path='search_exp')

    # def create(self, request, *args, **kwargs):

    #     if request.user.id == None:
    #         return Response(status=status.HTTP_401_UNAUTHORIZED)
    #     serializer = FaceSerializer(data=request.data)
    #     if serializer.is_valid():
    #         new_face = serializer.save()
    #         new_face.expcreatorid = request.user
    #         new_face.save()
    #         serializer = FaceSerializer(new_face, many=False)
    #         return Response(serializer.data, status=status.HTTP_201_CREATED)
    #     return JsonResponse({
    #         'message': 'Create a new face unsuccessful!'
    #     }, status=status.HTTP_400_BAD_REQUEST)

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
        path = f"./media/faces/{new_name}/"
        if not os.path.exists(path):
            os.makedirs(path)
        with open(path + file_obj.name, 'wb+') as destination:
            for chunk in file_obj.chunks():
                destination.write(chunk)
        faces = get_faces(path+file_obj.name)
        l_face = len(faces)
        if  l_face == 1:
            face_rec = ArcFace.ArcFace()
            x1= faces[0]['x1']
            y1= faces[0]['y1']
            x2= faces[0]['x2']
            y2= faces[0]['y2']
            emb1 = face_rec.calc_emb(cv2.imread(path+file_obj.name)[y1:y2,x1:x2])
            bf = emb1.tobytes()
            new_face = Face()
            new_face.name = person_name
            new_face.image_path = path+file_obj.name
            new_face.x1 = x1
            new_face.y1 = y1
            new_face.x2 = x2
            new_face.y2 = y2
            new_face.creatorID = User.objects.get(pk = request.user.id)
            new_face.emb = bf
            new_face.save()
            response = {
            'status': 'success',
            'code': status.HTTP_201_CREATED,
            'message': 'image uploaded successfully',
            'data': FaceSerializer(new_face,many=False).data
            }
            face_save = cv2.imread(path+file_obj.name)[y1:y2,x1:x2]
            os.remove(path+file_obj.name)
            cv2.imwrite(filename=path+str(new_face.Face_id)+'.png',img=face_save)
            new_face.image_path = path+str(new_face.Face_id)+'.png'
            new_face.save()
        elif l_face == 0 : 
            # The uploaded image doesn't contain a face, so we reject it
            os.remove(path+file_obj.name)
            response = {
            'status': 'success',
            'code': status.HTTP_400_BAD_REQUEST,
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

# def get_faces(image_path):
#     print('image_path',image_path)
#     detector = cv2.FaceDetectorYN.create("./weights/face_detection_yunet_2022mar.onnx", "", (320, 320))
#     # Read image
#     img = cv2.imread(image_path)
#     # Get image shape
#     img_W = int(img.shape[1])
#     img_H = int(img.shape[0])
#     # Set input size
#     detector.setInputSize((img_W, img_H))
#     # Getting detections
#     detections = detector.detect(img)
#     print(detections)
#     faces = detections[1]
#     try:
#         print(len(faces))
#     except:
#         faces = []
#     return faces

def get_faces(image_path):
    detector = RetinaFace(quality="normal")
    rgb_image = detector.read(image_path)
    faces = detector.predict(rgb_image)
    return faces

# def get_faces_by_img(image):
#     detector = cv2.FaceDetectorYN.create("./weights/face_detection_yunet_2022mar.onnx", "", (320, 320))
#     # Read image
#     img = image
#     # Get image shape
#     img_W = int(img.shape[1])
#     img_H = int(img.shape[0])
#     # Set input size
#     detector.setInputSize((img_W, img_H))
#     # Getting detections
#     detections = detector.detect(img)
#     faces = detections[1]
#     try:
#         print(len(faces))
#     except:
#         faces = []
#     return faces

def get_faces_by_img(image):
    detector = RetinaFace(quality="normal")
    rgb_image  = cv2.cvtColor(image,cv2.COLOR_BGR2RGB).astype(np.float32)
    faces = detector.predict(rgb_image)
    return faces


def crawl_Web_URL(url):
    # embeder = ArcFace.ArcFace()
    # detector = cv2.FaceDetectorYN.create("./weights/face_detection_yunet_2022mar.onnx", "", (320, 320))

    images = []
    image_urls = []
    image_raw = []

    try:
        url_txt = requests.get(url).text
        mysoup = BeautifulSoup(url_txt, 'html.parser')
        images = mysoup.find_all('img')
    except:
        return

    for image in images:
        try:
            # In image tag ,searching for "data-srcset"
            image_link = image["data-srcset"]

        # then we will search for "data-src" in img
        # tag and so on..
        except:
            try:
                # In image tag ,searching for "data-src"
                image_link = image["data-src"]
            except:
                try:
                    # In image tag ,searching for "data-fallback-src"
                    image_link = image["data-fallback-src"]
                except:
                    try:
                        # In image tag ,searching for "src"
                        image_link = image["src"]

                    # if no Source URL found
                    except:
                        continue
        try:
            image_link =  requests.compat.urljoin(url, image_link)
            
            r = requests.get(image_link).content
            arr = np.asarray(bytearray(r), dtype=np.uint8)
            out_img = cv2.imdecode(arr, -1)
            if out_img.shape[2] == 4:
                continue
            width = out_img.shape[1]
            height = out_img.shape[0] # keep original height

            if width < 50 and height < 50:
                continue

            image_urls.append(image_link)
            image_raw.append(out_img)

        except:
            continue
    return image_urls,image_raw

def cosine_distance(v1, v2):
    dot_product = np.dot(v1, v2)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    return dot_product / (norm_v1 * norm_v2)