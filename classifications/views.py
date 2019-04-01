from django.shortcuts import render
from rest_framework.response import Response
from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import AllowAny
from rest_framework.status import (
        HTTP_200_OK,
        HTTP_400_BAD_REQUEST,
    )

@api_view(['POST'])
@permission_classes((AllowAny,))
def classify(request):


    return Response({
        'class': 'Tiger',
        'accuracy': 0.84
    }, status=HTTP_200_OK)