from rest_framework.response import Response
from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import AllowAny
from rest_framework.status import (
        HTTP_200_OK,
        HTTP_400_BAD_REQUEST,
    )

from .classification_research.CNN import CNN

CNN_obj = CNN()


@api_view(['POST'])
@permission_classes((AllowAny,))
def classify(request):
    try:
        file = request.data['file']
    except KeyError:
        return Response({'file': ['no file']}, status=HTTP_400_BAD_REQUEST)

    result = CNN_obj.prediction(file)

    return Response(result, status=HTTP_200_OK)