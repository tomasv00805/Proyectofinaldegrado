"""
Función Lambda para invocar endpoint de SageMaker
Expone el modelo de ECG a través de API Gateway sin exponer credenciales AWS
"""

import json
import os
import boto3
import logging
from botocore.exceptions import ClientError

# Configurar logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Cliente de SageMaker Runtime (se inicializa en el contexto de Lambda)
sagemaker_runtime = None

def get_sagemaker_client():
    """
    Inicializa el cliente de SageMaker Runtime (singleton)
    
    Nota: AWS_REGION está disponible automáticamente en Lambda,
    no necesita configurarse como variable de entorno.
    """
    global sagemaker_runtime
    if sagemaker_runtime is None:
        # AWS_REGION está disponible automáticamente en Lambda
        # Si no está disponible, boto3 usará la región de Lambda automáticamente
        region = os.environ.get('AWS_REGION')  # None si no está disponible
        if region:
            sagemaker_runtime = boto3.client('sagemaker-runtime', region_name=region)
        else:
            # Si no hay AWS_REGION, boto3 detectará la región automáticamente desde Lambda
            sagemaker_runtime = boto3.client('sagemaker-runtime')
    return sagemaker_runtime

def lambda_handler(event, context):
    """
    Handler principal de Lambda
    
    Espera:
    - event["body"]: JSON string con formato {"signals": [[[...]]]}
    
    Retorna:
    - statusCode: 200 si éxito, 400/500 si error
    - body: JSON con respuesta del modelo o mensaje de error
    - headers: CORS habilitado
    """
    
    # Headers CORS por defecto
    cors_headers = {
        'Content-Type': 'application/json',
        'Access-Control-Allow-Origin': '*',  # En producción, usar dominio específico
        'Access-Control-Allow-Methods': 'POST, OPTIONS',
        'Access-Control-Allow-Headers': 'Content-Type, Authorization'
    }
    
    # Manejar preflight OPTIONS request
    if event.get('httpMethod') == 'OPTIONS' or event.get('requestContext', {}).get('http', {}).get('method') == 'OPTIONS':
        return {
            'statusCode': 200,
            'headers': cors_headers,
            'body': json.dumps({'message': 'OK'})
        }
    
    try:
        # Obtener nombre del endpoint desde variable de entorno
        endpoint_name = os.environ.get('SAGEMAKER_ENDPOINT')
        if not endpoint_name:
            logger.error("SAGEMAKER_ENDPOINT no está configurado")
            return {
                'statusCode': 500,
                'headers': cors_headers,
                'body': json.dumps({
                    'error': 'Configuración incorrecta: SAGEMAKER_ENDPOINT no definido'
                })
            }
        
        # Parsear body del request
        body = event.get('body')
        if not body:
            return {
                'statusCode': 400,
                'headers': cors_headers,
                'body': json.dumps({
                    'error': 'Body vacío. Se espera JSON con formato: {"signals": [[[...]]]}'
                })
            }
        
        # Si body es string, parsearlo
        if isinstance(body, str):
            try:
                request_data = json.loads(body)
            except json.JSONDecodeError as e:
                logger.error(f"Error parseando JSON: {e}")
                return {
                    'statusCode': 400,
                    'headers': cors_headers,
                    'body': json.dumps({
                        'error': f'JSON inválido: {str(e)}'
                    })
                }
        else:
            request_data = body
        
        # Validar estructura básica
        if not isinstance(request_data, dict):
            return {
                'statusCode': 400,
                'headers': cors_headers,
                'body': json.dumps({
                    'error': 'Body debe ser un objeto JSON con campo "signals"'
                })
            }
        
        if 'signals' not in request_data:
            return {
                'statusCode': 400,
                'headers': cors_headers,
                'body': json.dumps({
                    'error': 'Campo "signals" no encontrado en el request'
                })
            }
        
        # Validar que signals sea una lista
        signals = request_data['signals']
        if not isinstance(signals, list):
            return {
                'statusCode': 400,
                'headers': cors_headers,
                'body': json.dumps({
                    'error': 'Campo "signals" debe ser una lista'
                })
            }
        
        # Validar forma esperada: [batch, seq_len, n_channels] = [1, 2000, 3]
        if len(signals) == 0:
            return {
                'statusCode': 400,
                'headers': cors_headers,
                'body': json.dumps({
                    'error': 'Campo "signals" está vacío. Se espera al menos un ECG con forma [1, 2000, 3]'
                })
            }
        
        # Validar primer ECG (batch)
        first_ecg = signals[0]
        if not isinstance(first_ecg, list):
            return {
                'statusCode': 400,
                'headers': cors_headers,
                'body': json.dumps({
                    'error': 'Cada ECG en "signals" debe ser una lista (muestras temporales)'
                })
            }
        
        # Validar dimensiones esperadas
        seq_len = len(first_ecg)
        if seq_len != 2000:
            return {
                'statusCode': 400,
                'headers': cors_headers,
                'body': json.dumps({
                    'error': f'El ECG debe tener 2000 muestras temporales, pero se recibieron {seq_len}. Forma esperada: [1, 2000, 3]'
                })
            }
        
        # Validar número de canales
        if len(first_ecg) > 0:
            first_sample = first_ecg[0]
            if not isinstance(first_sample, list):
                return {
                    'statusCode': 400,
                    'headers': cors_headers,
                    'body': json.dumps({
                        'error': 'Cada muestra debe ser una lista de 3 valores (canales)'
                    })
                }
            
            n_channels = len(first_sample)
            if n_channels != 3:
                return {
                    'statusCode': 400,
                    'headers': cors_headers,
                    'body': json.dumps({
                        'error': f'El ECG debe tener 3 canales, pero se recibieron {n_channels}. Forma esperada: [1, 2000, 3]'
                    })
                }
        
        # Preparar payload para SageMaker (el formato exacto que espera el endpoint)
        payload = {
            'signals': signals
        }
        
        # Convertir a JSON string para SageMaker
        payload_json = json.dumps(payload, ensure_ascii=False)
        
        logger.info(f"Invocando endpoint: {endpoint_name}")
        logger.info(f"Payload size: {len(payload_json)} bytes")
        
        # Invocar endpoint de SageMaker
        try:
            client = get_sagemaker_client()
            response = client.invoke_endpoint(
                EndpointName=endpoint_name,
                ContentType='application/json',
                Body=payload_json.encode('utf-8')
            )
            
            # Leer respuesta
            response_body = response['Body'].read().decode('utf-8')
            
            # Parsear respuesta del modelo
            model_response = json.loads(response_body)
            
            logger.info(f"Respuesta del modelo: {model_response}")
            
            # Retornar respuesta exitosa
            return {
                'statusCode': 200,
                'headers': cors_headers,
                'body': json.dumps(model_response)
            }
            
        except ClientError as e:
            error_code = e.response.get('Error', {}).get('Code', 'Unknown')
            error_message = e.response.get('Error', {}).get('Message', str(e))
            
            logger.error(f"Error de SageMaker: {error_code} - {error_message}")
            
            return {
                'statusCode': 500,
                'headers': cors_headers,
                'body': json.dumps({
                    'error': f'Error invocando endpoint de SageMaker: {error_code}',
                    'message': error_message
                })
            }
        
        except json.JSONDecodeError as e:
            logger.error(f"Error parseando respuesta de SageMaker: {e}")
            return {
                'statusCode': 500,
                'headers': cors_headers,
                'body': json.dumps({
                    'error': 'Error procesando respuesta del modelo',
                    'message': str(e)
                })
            }
    
    except Exception as e:
        logger.error(f"Error inesperado: {str(e)}", exc_info=True)
        return {
            'statusCode': 500,
            'headers': cors_headers,
            'body': json.dumps({
                'error': 'Error interno del servidor',
                'message': str(e)
            })
        }

