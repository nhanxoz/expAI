# Generated by Django 4.1.3 on 2023-04-27 00:29

from django.conf import settings
from django.db import migrations, models
import django.db.models.deletion
import django.utils.timezone


class Migration(migrations.Migration):

    dependencies = [
        ('expAI', '0016_model_trained_model_trainedexpid_models_modelowner'),
    ]

    operations = [
        migrations.CreateModel(
            name='Face',
            fields=[
                ('Face_id', models.AutoField(db_column='faceID', primary_key=True, serialize=False)),
                ('image_path', models.CharField(blank=True, db_column='imagePath', max_length=2000)),
                ('name', models.CharField(blank=True, db_column='name', max_length=500, null=True)),
                ('infor', models.CharField(blank=True, db_column='infor', max_length=1000, null=True)),
                ('x1', models.FloatField(default=0)),
                ('y1', models.FloatField(default=0)),
                ('x2', models.FloatField(default=0)),
                ('y2', models.FloatField(default=0)),
                ('emb', models.BinaryField(max_length=2048)),
                ('time', models.DateTimeField(default=django.utils.timezone.now)),
                ('creatorID', models.ForeignKey(db_column='faceCreatorID', on_delete=django.db.models.deletion.CASCADE, to=settings.AUTH_USER_MODEL)),
            ],
            options={
                'db_table': 'Face',
                'managed': True,
            },
        ),
    ]
