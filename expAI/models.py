# This is an auto-generated Django model module.
# You'll have to do the following manually to clean this up:
#   * Rearrange models' order
#   * Make sure each model has one field with primary_key=True
#   * Make sure each ForeignKey and OneToOneField has `on_delete` set to the desired behavior
#   * Remove `managed = True` lines if you wish to allow Django to create, modify, and delete the table
# Feel free to rename the models, but don't rename db_table values or field names.
from django.db import models
from django.utils import timezone
from django.contrib.auth.models import BaseUserManager, AbstractBaseUser, PermissionsMixin


class UserManager(BaseUserManager):
    def _create_user(self, email, password, is_staff , is_superuser, **extra_fields):
        if not email:
            raise ValueError('Users must have an email address')

        now = timezone.now()
        user = self.model(
            email=self.normalize_email(email),
            is_staff=is_staff,

            is_active=False,
            is_superuser=is_superuser,
            last_login=now,
            joined_at=now,
            **extra_fields
        )
        user.set_password(password)
        user.save(using=self._db)
        return user

    def get_by_natural_key(self, username):
        return self.get(**{'{}__iexact'.format(self.model.USERNAME_FIELD): username})

    def create_user(self, email, password, **extra_fields):
        role = Roles.objects.get(rolename="STUDENT")
        return self._create_user(email, password, is_superuser= False, is_staff=False,  **extra_fields)

    def create_superuser(self, email, password, **extra_fields):
        return self._create_user(email, password, True, True, **extra_fields)


class User(AbstractBaseUser, PermissionsMixin):
    email = models.EmailField('Email', max_length=255, unique=True)
    name = models.CharField('Name', max_length=255, blank=True)
    is_staff = models.BooleanField('Is staff', default=False)
    is_active = models.BooleanField('Is active', default=True)
    joined_at = models.DateTimeField('Joined at', default=timezone.now)
    roleid = models.ForeignKey('roles', models.DO_NOTHING, db_column='roleid', blank=True, null=True)
    usrfullname = models.CharField(db_column='usrFullName', max_length=50  , blank=True, null=True)  # Field name made lowercase.
    usrdob = models.DateField(db_column='usrDoB', blank=True, null=True)  # Field name made lowercase.
    usrfaculty = models.CharField(db_column='usrFaculty', max_length=45, blank=True, null=True)  # Field name made lowercase.
    chose_class = models.BooleanField('User chose class', default= False)
    objects = UserManager()

    USERNAME_FIELD = 'email'


    # def __str__(self):
    #     return str(self.pk)

    class Meta:
        verbose_name = 'User'
        verbose_name_plural = 'Users'

    def get_full_name(self):
        return self.usrfullname

    def get_short_name(self):
        return self.name

class Datasets(models.Model):
    datasetid = models.AutoField(db_column='datasetID', primary_key=True)  # Field name made lowercase.
    datasetname = models.CharField(db_column='datasetName', max_length=100, blank=True, null=True)  # Field name made lowercase.
    datasettype = models.ForeignKey("TypePermission", models.DO_NOTHING,db_column='datasetType', blank=True, null=True)  # Field name made lowercase.
    datasetsoftID = models.ForeignKey("SoftwareLibs", models.DO_NOTHING,db_column='datasetsoftID', blank=True, null=True)  # Field name made lowercase.
    datasetfolderurl = models.CharField(db_column='datasetFolderURL', max_length=200, blank=True, null=True)  # Field name made lowercase.
    datasetsum = models.IntegerField(db_column='datasetSum', blank=True, null=True)  # Field name made lowercase.
    datasetcreatedtime = models.DateTimeField(db_column='datasetCreatedTime', auto_now_add=True, blank=True)  # Field name made lowercase.
    datasetdescription = models.CharField(db_column='datasetDescription', max_length=200  , blank=True, null=True)  # Field name made lowercase.
    datasetowner = models.ForeignKey("User", models.DO_NOTHING, db_column='datasetOwner', blank=True, null=True)
    class Meta:
        managed = True
        db_table = 'datasets'


class Evaluations(models.Model):
    evaluateid = models.AutoField(db_column='evaluateID', primary_key=True)  # Field name made lowercase.
    evaluateconfusionmatrixtraining = models.CharField(db_column='evaluateConfusionMatrixTraining', max_length=45, blank=True, null=True)  # Field name made lowercase.
    evaluateconfusionmatrixtesting = models.CharField(db_column='evaluateConfusionMatrixTesting', max_length=45, blank=True, null=True)  # Field name made lowercase.
    evaluateconfutionmatrixvalidation = models.CharField(db_column='evaluateConfutionMatrixValidation', max_length=45, blank=True, null=True)  # Field name made lowercase.
    evaluatenumclass = models.IntegerField(db_column='evaluateNumClass', blank=True, null=True)  # Field name made lowercase.
    

    class Meta:
        managed = True
        db_table = 'evaluations'


class Experiments(models.Model):
    expname = models.CharField(db_column='expName', max_length=100  , blank=True, null=True)  # Field name made lowercase.
    expid = models.AutoField(db_column='expID', primary_key=True)  # Field name made lowercase.
    expname = models.CharField(db_column='expName', max_length=100 , blank=True, null=True)  # Field name made lowercase.
    expcreatorid = models.ForeignKey('User', models.CASCADE, db_column='expCreatorID', blank=True, null=True)  # Field name made lowercase.
    expcreatedtime = models.DateTimeField(db_column='expCreatedTime', blank=True, null=True)  # Field name made lowercase.
    expmodelid = models.ForeignKey('Models', models.SET_NULL, db_column='expModelID', blank=True, null=True)  # Field name made lowercase.
    expdatasetid = models.ForeignKey(Datasets, models.SET_NULL, db_column='expDatasetID', blank=True, null=True)  # Field name made lowercase.
    expfilelog = models.CharField(db_column='expFileLog', max_length=100, blank=True, null=True)  # Field name made lowercase.
    expsoftwarelibid = models.ForeignKey('Softwarelibs', models.CASCADE, db_column='expSoftwareLibID', blank=True, null=True)  # Field name made lowercase.
    expaftertrainmodelpath = models.CharField(db_column='expAfterTrainModelPath', max_length=200, blank=True, null=True)  # Field name made lowercase.
    expstatus = models.IntegerField(db_column='expStatus',default=1)
    class Meta:
        managed = True
        db_table = 'experiments'


class Models(models.Model):
    modelid = models.AutoField(db_column='modelID', primary_key=True)  # Field name made lowercase.
    modelname = models.CharField(db_column='modelName', max_length=100, blank=True, null=True)  # Field name made lowercase.
    modeltype = models.ForeignKey("TypePermission", models.DO_NOTHING,db_column='datasetType', blank=True, null=True)  # Field name made lowercase.
    modelfiletutorial = models.CharField(db_column='modelFIleTutorial', max_length=200, blank=True, null=True)  # Field name made lowercase.
    modelfiledescription = models.CharField(db_column='modelFileDescription', max_length=200, blank=True, null=True)  # Field name made lowercase.
    modeldescription = models.CharField(db_column='modelDescription', max_length=45, blank=True, null=True)  # Field name made lowercase.
    modeleventtype = models.CharField(db_column='modelEventType', max_length=45, blank=True, null=True)  # Field name made lowercase.
    modelcreator = models.CharField(db_column='modelCreator', max_length=20, blank=True, null=True)  # Field name made lowercase.
    modelcreatedtime = models.DateTimeField(db_column='modelCreatedTime', blank=True, null=True)  # Field name made lowercase.
    modelsoftlibid = models.IntegerField(db_column='modelSoftLibID', blank=True, null=True)  # Field name made lowercase.
    pretrainpath = models.CharField(db_column='pretrainpath',max_length=1000,null=True,blank=True)
    default_json_Paramsconfigs = models.CharField(db_column='jsonStringParams', max_length=2500, blank=True, null=True)

    class Meta:
        managed = True
        db_table = 'models'


class Objectembeddings(models.Model):
    objid = models.OneToOneField('Objects', models.DO_NOTHING, db_column='objID', primary_key=True)  # Field name made lowercase.
    expid = models.ForeignKey(Experiments, models.DO_NOTHING, db_column='expID')  # Field name made lowercase.
    note = models.CharField(max_length=100, blank=True, null=True)

    class Meta:
        managed = True
        db_table = 'objectembeddings'
        unique_together = (('objid', 'expid'),)


class Objects(models.Model):
    objid = models.AutoField(db_column='objID', primary_key=True)  # Field name made lowercase.
    objname = models.CharField(db_column='objName', max_length=50  , blank=True, null=True)  # Field name made lowercase.
    objgeneralinfo = models.CharField(db_column='objGeneralInfo', max_length=500, blank=True, null=True)  # Field name made lowercase.
    objurlfolder = models.CharField(db_column='objURLFolder', max_length=200, blank=True, null=True)  # Field name made lowercase.
    objcreatedtime = models.DateTimeField(db_column='objCreatedTime', blank=True, null=True)  # Field name made lowercase.
    objcreator = models.CharField(db_column='objCreator', max_length=20, blank=True, null=True)  # Field name made lowercase.
    objtype = models.CharField(db_column='objType', max_length=45, blank=True, null=True)  # Field name made lowercase.

    class Meta:
        managed = True
        db_table = 'objects'


class Paramsconfigs(models.Model):
    configid = models.AutoField(db_column='configID', primary_key=True)  # Field name made lowercase.
    jsonstringparams = models.CharField(db_column='jsonStringParams', max_length=2500, blank=True, null=True)
    trainningstatus = models.IntegerField(db_column='trainningstatus',default=1)
    configimagesize = models.CharField(db_column='configImageSize', max_length=45, blank=True, null=True)  # Field name made lowercase.
    configlearningrate = models.FloatField(db_column='configLearningRate', blank=True, null=True)  # Field name made lowercase.
    configalgorithm = models.CharField(db_column='configAlgorithm', max_length=45, blank=True, null=True)  # Field name made lowercase.
    configepoch = models.IntegerField(db_column='configEpoch', blank=True, null=True)  # Field name made lowercase.
    configbatchsize = models.IntegerField(db_column='configBatchSize', blank=True, null=True)  # Field name made lowercase.
    configexpid = models.ForeignKey(Experiments, models.CASCADE, db_column='configExpID', blank=True, null=True)  # Field name made lowercase.
    configresid = models.CharField(db_column='configResID', max_length=20, blank=True, null=True)  # Field name made lowercase.
    configevaluateid = models.ForeignKey(Evaluations, models.CASCADE, db_column='configEvaluateID', blank=True, null=True)  # Field name made lowercase.
    configaftertrainmodelpath = models.CharField(db_column='expAfterTrainModelPath', max_length=200, blank=True, null=True)  # Field name made lowercase.
    configfilelog = models.CharField(db_column='expFileLog', max_length=100, blank=True, null=True)


    class Meta:
        managed = True
        db_table = 'paramsconfigs'

class Trainningresults(models.Model):
    trainresultid =  models.AutoField(db_column='trainResultID', primary_key=True)
    trainresultindex = models.IntegerField(db_column='trainResultIndex',default=0)
    lossvalue = models.FloatField(db_column='lossvalue')
    accuracy = models.FloatField(db_column='accuracy')
    configid = models.ForeignKey(Paramsconfigs, models.CASCADE, db_column='configID', blank=True, null=True)  # Field name made lowercase.
    is_last = models.BooleanField(db_column='is_last',null=True,blank=True,default=False)

class Results(models.Model):
    resultid = models.AutoField(db_column='resultID', primary_key=True)  # Field name made lowercase.
    resulttestingdataset = models.ForeignKey(Datasets,models.SET_NULL, db_column='resultTestingDataset', blank=True, null=True)  # Field name made lowercase.
    resultaccuracy = models.FloatField(db_column='resultAccuracy', blank=True, null=True)  # Field name made lowercase.
    resultdetail = models.CharField(db_column='resultDetail', max_length=800, blank=True, null=True)  # Field name made lowercase.
    resultconfigid = models.ForeignKey(Paramsconfigs, models.CASCADE, db_column='resultConfigID', blank=True, null=True)  # Field name made lowercase.

    class Meta:
        managed = True
        db_table = 'results'

class Predict(models.Model):
    predictid = models.AutoField(db_column='predictId',primary_key=True)
    inputpath = models.CharField(db_column='inputPath',max_length=800,null=True,blank=True)
    inputpath2 = models.CharField(db_column='inputPath2',max_length=800,null=True,blank=True)
    outputpath = models.CharField(db_column='outputPath',max_length=800,null=True,blank=True)
    datatype = models.CharField(db_column='datatype',max_length=200,null=True,blank=True)

    accuracy = models.FloatField(db_column='accuracy',null=True,blank=True)
    details = models.CharField(db_column='details',max_length=800,null=True,blank=True)

    configid = models.ForeignKey(Paramsconfigs, models.CASCADE, db_column='resultConfigID', blank=True, null=True)  # Field name made lowercase.
    class Meta:
        managed = True
        db_table = 'predict'
        




class Roles(models.Model):
    roleid = models.AutoField(db_column='roleID', primary_key=True)  # Field name made lowercase.
    rolename = models.CharField(db_column='roleName', unique=True, max_length=45, blank=True, null=True)  # Field name made lowercase.

    class Meta:
        managed = True
        db_table = 'roles'


class Softwarelibs(models.Model):
    softwarelibid = models.AutoField(db_column='softwarelibID', primary_key=True)  # Field name made lowercase.
    softwarelibname = models.CharField(db_column='softwarelibName', max_length=45, blank=True, null=True)  # Field name made lowercase.
    softwareliburl = models.CharField(db_column='softwarelibURL', max_length=200, blank=True, null=True)  # Field name made lowercase.
    softwarelibdescription = models.CharField(db_column='softwarelibDescription', max_length=1000, blank=True, null=True)  # Field name made lowercase.

    class Meta:
        managed = True
        db_table = 'softwarelibs'

class Class(models.Model):
    classid = models.AutoField(db_column= "classID", primary_key=True)
    classcode = models.CharField(db_column="classCode", max_length=45, blank=True, null=True)
    classname = models.CharField(db_column="className", max_length=45, blank=True, null=True)
    classschoolyear=models.CharField(db_column="classSchoolYear", max_length=10, blank=True, null=True)
    class Meta:
        managed = True
        db_table = 'Class'

class TypePermission(models.Model):
    typeid = models.AutoField(primary_key=True)
    typename = models.CharField(db_column="typeName", max_length=20, blank=True, null=True)
    class Meta:
        managed = True
        db_table="TypePermission"

class ClassUser(models.Model):
    class_user_id = models.AutoField(db_column="classUserID", primary_key=True)
    class_id = models.ForeignKey("Class", models.DO_NOTHING, db_column="classid", blank=True, null=True )
    user_id = models.ForeignKey("User", models.DO_NOTHING, db_column="ID", blank=True, null=True)
    status = models.IntegerField(db_column='status', blank=True, null=True)
    time_regis = models.DateTimeField('register time', default=timezone.now)
    time_approve = models.DateTimeField('approve time', default=timezone.now)
    class Meta:
        managed = True
        db_table="ClassUser"

from django.dispatch import receiver
from django.urls import reverse
from django_rest_passwordreset.signals import reset_password_token_created
from django.core.mail import send_mail  


@receiver(reset_password_token_created)
def password_reset_token_created(sender, instance, reset_password_token, *args, **kwargs):

    email_plaintext_message = "{}?token={}".format(reverse('password_reset:reset-password-request'), reset_password_token.key)
    # 1EmailBackend
    # aiexperimentbe@gmail.com
    send_mail(
        # title:
        "Password Reset for {title}".format(title="Some website title"),
        # message:
        email_plaintext_message,
        # from:
        "aiexperimentbe@gmail.com",
        # to:
        [reset_password_token.user.email]
    )