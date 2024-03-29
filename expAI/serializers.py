from rest_framework.serializers import ModelSerializer
from .models import *
from django.contrib.auth import authenticate
from rest_framework import serializers
from .validators import validate_username
from .permissions import *

class LoginSerializer(serializers.Serializer):
    email = serializers.EmailField()
    password = serializers.CharField()

    def validate(self, attrs):
        user = authenticate(username=attrs['email'], password=attrs['password'])
        if not user:
            raise serializers.ValidationError('Email hoặc mật khẩu không đúng')

        if not user.is_active:
            raise serializers.ValidationError('Đang chờ phê duyệt')

        return {'user': user}


class ChangePasswordSerializer(serializers.Serializer):
    model = User

    """
    Serializer for password change endpoint.
    """
    old_password = serializers.CharField(required=True)
    new_password = serializers.CharField(required=True)
class ChangeNameSerializer(serializers.Serializer):
    model = User

    """
    Serializer for password change endpoint.
    """
    # password = serializers.CharField(required=True)
    name = serializers.CharField(required=True)
    # usrclass = serializers.ListField(required=True)
    usrfullname = serializers.CharField(required=True)
    usrdob = serializers.DateField(required=True)
    usrfaculty = serializers.CharField(required=True)
class DestroyUserSerializer(serializers.Serializer):
    model = User

    """
    Serializer for password change endpoint.
    """
    password = serializers.CharField(required=True)

class ChangePassword2Serializer(serializers.Serializer):
    model = User

    """
    Serializer for password change endpoint.
    """
    id_user = serializers.CharField(required=True)
    new_password = serializers.CharField(required=True)



class User2Serializer(serializers.ModelSerializer):

    class Meta:
        model = User
        fields = (
            'id',
            'last_login',
            'email',
            'name',
            'is_active',
            'joined_at',
            'usrclass',
            'is_staff',
            'roleid',
            'usrfullname',
            'usrdob',
            'usrfaculty',
            'chose_class'
        )
        read_only_fields = ( 'is_staff','last_login', 'is_active', 'joined_at','chose_class', 'usrclass')
        extra_kwargs = {
            'name': {'required': True}
        }

    # @staticmethod
    # def validate_email(value):
    #     return validate_username(value)

    def create(self, validated_data):
        return User.objects.create_user(
                    validated_data.pop('email'),
                    validated_data.pop('password'),
                    **validated_data
                )
class UserCreateSerializer(serializers.ModelSerializer):
    password = serializers.CharField(write_only=True)

    class Meta:
        model = User
        fields = ('email', 'password', 'name', 'roleid', 'usrfullname', 'usrdob', 'usrfaculty', 'chose_class', 'usrclass',)

    def create(self, validated_data):
        password = validated_data.pop('password')
        user = User.objects.create_user(password=password, **validated_data)
        return user
class UserSerializer(serializers.ModelSerializer):

    class Meta:
        model = User
        fields = (
            'id',
            'last_login',
            'email',
            'name',
            'is_active',
            'joined_at',
            'password',
            'is_staff',
            'roleid',
            'usrclass',
            'usrfullname',
            'usrdob',
            'usrfaculty',
            'chose_class'
        )
        read_only_fields = ( 'is_staff','last_login', 'is_active', 'joined_at','chose_class')
        extra_kwargs = {
            'password': {'required': True, 'write_only': True},
            'name': {'required': True}
        }

    # @staticmethod
    # def validate_email(value):
    #     return validate_username(value)

    def create(self, validated_data):
        return User.objects.create_user(
                    validated_data.pop('email'),
                    validated_data.pop('password'),
                    **validated_data
                )

class SoftwareLibsSerializer(ModelSerializer):
    class Meta:
    #         softwarelibid = models.CharField(db_column='softwarelibID', primary_key=True, max_length=20)  # Field name made lowercase.
    # softwarelibname = models.CharField(db_column='softwarelibName', max_length=45, blank=True, null=True)  # Field name made lowercase.
    # softwareliburl = models.CharField(db_column='softwarelibURL', max_length=200, blank=True, null=True)  # Field name made lowercase.
    # softwarelibdescription = models.CharField(db_column='softwarelibDescription', max_length=1000, blank=True, null=True)  # Field name made lowercase.

        model = Softwarelibs
        fields = '__all__'


class ExperimentsSerializer(ModelSerializer):
    class Meta:
        model = Experiments
        fields = '__all__'
        read_only_fields = ('expcreatorid',)


class DatasetsSerializer(ModelSerializer):
    datasetowner_name = serializers.ReadOnlyField(source='get_datasetowner_name')

    class Meta:
        model = Datasets
        fields = ('datasetid', 'datasetname', 'datasettype', 'datasetsoftID', 'datasetfolderurl', 'datasetsum',
                  'datasetcreatedtime', 'datasetdescription', 'datasetowner', 'datasetowner_name')
        read_only_fields = ('datasetowner_name',)

    def create(self, validated_data):
        datasetowner_id = validated_data.pop('datasetowner_id', None)
        instance = super().create(validated_data)
        if datasetowner_id:
            instance.set_datasetowner_id(datasetowner_id)
            instance.save()
        return instance
class ParamsconfigsSerializer(ModelSerializer):
    class Meta:
        model = Paramsconfigs
        fields = '__all__'
class ResultsSerializer(ModelSerializer):
    resultconfig = ParamsconfigsSerializer( read_only=True, many=False)
    resultdataset = DatasetsSerializer( read_only=True, many=False)
    class Meta:
        model = Results
        fields = '__all__'
        read_only_fields = ('resultaccuracy','resultdetail',)
class ModelsSerializer(ModelSerializer):
    class Meta:
        model = Models
        fields = '__all__'

class Model_trainedSerializer(ModelSerializer):
    class Meta:
        model = Model_trained
        depth = 1
        fields = '__all__'

class Creat_Model_trainedSerializer(ModelSerializer):
    class Meta:
        model = Model_trained
        fields = '__all__'
        
class TrainningresultsSerializer(ModelSerializer):
    class Meta:
        model = Trainningresults
        fields = '__all__'
class ClassesSerializer(ModelSerializer):
    class Meta:
        model = Class
        fields = '__all__'

class PredictSerializer(ModelSerializer):
    class Meta:
        model = Predict
        fields = '__all__'
        read_only_fields = ('accuracy','details','outputpath',)

class RequestToClassSerializer(serializers.Serializer):
    id_user = serializers.IntegerField(required=True)
    id_class = serializers.IntegerField(required=True)
class DenyToClassSerializer(serializers.Serializer):

    id_user_class = serializers.IntegerField(required=True)
class ThongkeRole(serializers.Serializer):

    admin = serializers.IntegerField()
    hocvien = serializers.IntegerField()
    giaovien = serializers.IntegerField()
class AssignToClassSerializer(serializers.Serializer):
    id_user = serializers.IntegerField(required=True)
    id_class = serializers.IntegerField(required=True)


class ApproveToClassSerializer(serializers.Serializer):

    id_user_class = serializers.IntegerField(required=True)
    status = serializers.IntegerField(required=True)


class DisableAccountSerializer(serializers.Serializer):

    status = serializers.IntegerField(required=True)
class UserClassSerializer(ModelSerializer):
    class Meta:
        model = ClassUser
        fields = '__all__'
class UserClass2Serializer(ModelSerializer):
    class Meta:
        model = ClassUser
        fields = '__all__'
        depth = 1
class ClassUserSerializer(serializers.ModelSerializer):
    id = serializers.ReadOnlyField(source='user_id.id')
    email = serializers.ReadOnlyField(source='user_id.email')
    name = serializers.ReadOnlyField(source='user_id.name')
    rolename = serializers.ReadOnlyField(source='user_id.roleid.rolename')
    usrfullname = serializers.ReadOnlyField(source='user_id.usrfullname')
    usrdob = serializers.ReadOnlyField(source='user_id.usrdob')
    usrfaculty = serializers.ReadOnlyField(source='user_id.usrfaculty')

    class Meta:
        model = ClassUser
        fields = ['class_user_id', 'class_id', 'status', 'time_regis', 'time_approve', 'id', 'email', 'name', 'rolename', 'usrfullname', 'usrdob', 'usrfaculty']

class ConfirmUserSerializer(serializers.Serializer):
    id_user =serializers.IntegerField(required=True)
    status = serializers.BooleanField(required=True)

class FaceSerializer(ModelSerializer):
    class Meta:
        model = Face
        fields = '__all__'