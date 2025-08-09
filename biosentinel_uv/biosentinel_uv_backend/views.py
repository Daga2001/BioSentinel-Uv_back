from rest_framework.decorators import api_view, authentication_classes, permission_classes
from rest_framework.permissions import IsAuthenticated, AllowAny
from rest_framework.authentication import TokenAuthentication
from rest_framework.response import Response
from rest_framework import status
import os
import openai
import time
import base64
import os
from django.utils import timezone

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
            "success": False,
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
            "success": False,
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
            "success": False,
            "message": f"Ocurrió un error al generar el informe: {str(e)}"
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

    return Response({
        "success": True,
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
        "resolution": 10, # resolución en metros (opcional, por defecto 10)
    }

    Returns:
    - 200 OK con:
        - classifications: { clase: { color, count } }
        - image: "<imagen en base64>"
    """
    data = request.data
    modelo = data.get("model")

    # Inicializa el tiempo de procesamiento
    start_time = time.time()

    # === 🔍 SEGMENTACIÓN SEGÚN MODELO ===
    try:
        if modelo == "segformer-b0-ade20k":
            result = utils.segmentar_con_segformer_b0(image_path)
        elif modelo == "sam":
            result = utils.segmentar_con_clipseg_sam(image_path)
        elif modelo == "k-means":
            if isinstance(image_path, list):
                image_path = utils.stack_band_tiffs_to_multiband(image_path)
            result = utils.segmentar_con_kmeans(image_path)
        elif modelo == "mkanet":
            if isinstance(image_path, list):
                image_path = utils.stack_band_tiffs_to_multiband(image_path)
            result = utils.segmentar_con_mkanet(image_path)
        else:
            print(f"Modelo '{modelo}' no soportado aún.")
            return Response({
                "success": False,
                "message": f"Modelo '{modelo}' no soportado aún."
            }, status=status.HTTP_400_BAD_REQUEST)
    except Exception as e:
        print("❌ Error al utilizar el modelo de IA:", str(e))
        return Response({
            "success": False,
            "message": f"Error durante segmentación: {str(e)}"
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
    print("✅ Segmentación completada. Resultado:", result.keys())
    
    # === ⏱️ Calcular tiempo de procesamiento ===
    processing_time_ms = int((time.time() - start_time) * 1000)

    # === 📸 Codifica imagen en base64 ===
    try:
        with open(result["image_path"], "rb") as image_file:
            image_base64 = base64.b64encode(image_file.read()).decode('utf-8')
    except Exception as e:
        return Response({
            "success": False,
            "message": f"No se pudo codificar la imagen segmentada: {str(e)}"
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

    # timestamp — marca de tiempo de la segmentación
    # Esto es simplemente la hora actual en UTC en formato ISO 8601:
    timestamp_iso = timezone.now().isoformat().replace("+00:00", "Z")

    # Total de píxeles procesados
    total_pixels = sum(c["count"] for c in result["classifications"].values())

    return Response({
        "success": True,
        "model": modelo,
        "resolution": res,
        "classifications": result["classifications"],
        "overlayImage": image_base64,
        "metadata": {
            "categories": len(result["classifications"]),
            "processingTime": processing_time_ms,
            "timestamp": timestamp_iso,
            "totalPixels": total_pixels
        }
    }, status=status.HTTP_200_OK)   
    
    
@api_view(['POST'])
@authentication_classes([TokenAuthentication])
@permission_classes([AllowAny])
def generar_segmentacion_bs10(request):
    data = request.data
    model = data.get("model")
    taxon  = data.get("taxon")
    lon = data.get("longitude")
    lat = data.get("latitude")
    radius_km = data.get("radius_km", 50)
    if lon is None or lat is None:
        return Response({
            "success": False,
            "message": "Se requiere 'longitude' y 'latitude' en el body para el modelo bs1.0."
        }, status=status.HTTP_400_BAD_REQUEST)
    geojson_path = utils.run_bs1_model(lon, lat, radius_km, taxon)
    # Obtiene el archivo GeoJSON directamente
    with open(geojson_path, "r", encoding="utf-8") as f:
        geojson_content = f.read()
    return Response({
        "success": True,
        "model": model,
        "geojson": geojson_content
    }, status=status.HTTP_200_OK)