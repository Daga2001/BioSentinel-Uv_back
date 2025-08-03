from rest_framework.decorators import api_view, authentication_classes, permission_classes
from rest_framework.permissions import IsAuthenticated, AllowAny
from rest_framework.authentication import TokenAuthentication
from rest_framework.response import Response
from rest_framework import status
import os
import openai

import base64
import os

from . import utils

# ===========================================================================================
# Vista para generar un informe ambiental con GPT-4.1-nano a partir de datos satelitales.
# Validaciones:
# - Solo usuarios autenticados pueden generar informes
# - Se requieren los datos de coberturas terrestres
# ===========================================================================================

@api_view(['POST'])
@authentication_classes([TokenAuthentication])
@permission_classes([AllowAny])
def generar_informe_ambiental(request):
    """
    Genera un informe ambiental con base en una distribución de coberturas terrestres.

    Body requerido (JSON):
    - pixels_per_class: {<class_id>: <pixel_count>}
    - class_names (opcional): {<class_id>: <nombre_clase>}

    Returns:
    - 200 OK con el informe generado
    - 400 Bad Request si falta información
    """
    data = request.data

    # Obtenemos las clases y pixeles por clase del cuerpo de la solicitud.
    pixels_per_class = data.get("pixels_per_class")
    class_names = data.get("class_names", {})

    if not pixels_per_class or not isinstance(pixels_per_class, dict):
        return Response({
            "error": True,
            "message": "Se requiere un diccionario válido de 'pixels_per_class'."
        }, status=status.HTTP_400_BAD_REQUEST)

    # Construir el prompt
    summary_prompt = "Eres un experto ambiental que analiza imágenes satelitales. Genera un informe basado en la siguiente distribución de coberturas terrestres:\n\n"
    for class_id, pixel_count in pixels_per_class.items():
        class_name = class_names.get(str(class_id), f"Clase_{class_id}")
        summary_prompt += f"- {class_name}: {pixel_count} píxeles\n"
    summary_prompt += "\nIncluye posibles implicaciones ecológicas, riesgos ambientales y sugerencias de conservación.\n"

    # Obtener la API key de entorno
    api_key = "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"
    if not api_key:
        return Response({
            "error": True,
            "message": "API key de OpenAI no configurada en el entorno."
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

    try:
        client = openai.OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model="gpt-4.1-nano",
            messages=[
                {"role": "system", "content": "Eres un experto ambiental que redacta informes técnicos claros y útiles."},
                {"role": "user", "content": summary_prompt}
            ],
            temperature=0.7,
            max_tokens=500,
        )
        report = response.choices[0].message.content.strip()
    except Exception as e:
        return Response({
            "error": True,
            "message": f"Ocurrió un error al generar el informe: {str(e)}"
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

    return Response({
        "error": False,
        "message": "Informe generado exitosamente.",
        "data": {
            "prompt_usado": summary_prompt,
            "reporte": report
        }
    }, status=status.HTTP_200_OK)

# ===========================================================================================
# Vista para generar una imagen segmentada a partir de un GeoJSON y un modelo especificado.
# ===========================================================================================

@api_view(['POST'])
@authentication_classes([TokenAuthentication])
@permission_classes([AllowAny])
def generar_segmentacion(request):
    """
    Genera una imagen segmentada a partir de un modelo IA y una región definida en GeoJSON.

    Body esperado:
    {
        "model": "GPTx",
        "geojson": { ... }
    }

    Returns:
    - 200 OK con:
        - classifications: { clase: { color, count } }
        - image: "<imagen en base64>"
    """
    data = request.data
    modelo = data.get("model")
    geojson = data.get("geojson")

    # Valida el cuerpo de la solicitud
    if not modelo or not geojson:
        return Response({
            "error": True,
            "message": "Se requiere 'model' y 'geojson' en el body."
        }, status=status.HTTP_400_BAD_REQUEST)

    # Intenta descargar la imagen desde el geojson de la solicitud
    try:
        image_path = utils.descargar_imagen_desde_geojson(geojson)
    except Exception as e:
        return Response({
            "error": True,
            "message": f"Error al descargar imagen: {str(e)}"
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

    # === 🔍 SEGMENTACIÓN SEGÚN MODELO ===
    try:
        if modelo == "SegFormer-B0 ADE20K":
            result = utils.segmentar_con_segformer_b0(image_path)
        else:
            return Response({
                "error": True,
                "message": f"Modelo '{modelo}' no soportado aún."
            }, status=status.HTTP_400_BAD_REQUEST)
    except Exception as e:
        return Response({
            "error": True,
            "message": f"Error durante segmentación: {str(e)}"
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

    # === 📸 Codifica imagen en base64 ===
    try:
        with open(result["image_path"], "rb") as image_file:
            image_base64 = base64.b64encode(image_file.read()).decode('utf-8')
    except Exception as e:
        return Response({
            "error": True,
            "message": f"No se pudo codificar la imagen segmentada: {str(e)}"
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

    return Response({
        "error": False,
        "classifications": result["classifications"],
        "image": image_base64
    }, status=status.HTTP_200_OK)   