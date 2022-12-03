import django_filters
from .models import *
class DatasetsFilter(django_filters.FilterSet):
    class Meta:
        model = Datasets
        fields = {
            'datasetname': ['contains'],
            'datasettype': ['exact'],
            'datasetproblem': ['exact'],
            'datasetfolderurl': ['contains'],
            'datasetdescription': ['contains'],
            
            
        }