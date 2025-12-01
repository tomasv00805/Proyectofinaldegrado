"""
Script para diagnosticar problemas con el endpoint de SageMaker
"""
import boto3
import json
from datetime import datetime, timedelta

# Configuración
ENDPOINT_NAME = "cnn1d-lstm-ecg-v1-serverless"
AWS_REGION = "us-east-1"

# -*- coding: utf-8 -*-
print("=" * 70)
print("[DIAGNOSTICO] DIAGNOSTICO DEL ENDPOINT")
print("=" * 70)

# 1. Verificar estado del endpoint
print("\n[1] Verificando estado del endpoint...")
sagemaker_client = boto3.client('sagemaker', region_name=AWS_REGION)

try:
    response = sagemaker_client.describe_endpoint(EndpointName=ENDPOINT_NAME)
    status = response['EndpointStatus']
    print(f"   Estado: {status}")
    
    if status == 'InService':
        print("   [OK] Endpoint esta activo")
    elif status == 'Creating':
        print("   [WAIT] Endpoint aun se esta creando...")
    elif status == 'Failed':
        print("   [ERROR] Endpoint fallo al crearse")
        if 'FailureReason' in response:
            print(f"   Razon: {response['FailureReason']}")
    else:
        print(f"   [WARN] Estado: {status}")
        
except Exception as e:
    print(f"   [ERROR] Error: {e}")

# 2. Ver logs de CloudWatch
print("\n[2] Revisando logs de CloudWatch...")
logs_client = boto3.client('logs', region_name=AWS_REGION)
log_group_name = f"/aws/sagemaker/Endpoints/{ENDPOINT_NAME}"

try:
    # Obtener logs de la última hora
    end_time = int(datetime.now().timestamp() * 1000)
    start_time = int((datetime.now() - timedelta(hours=2)).timestamp() * 1000)
    
    response = logs_client.filter_log_events(
        logGroupName=log_group_name,
        startTime=start_time,
        endTime=end_time,
        limit=100
    )
    
    events = response.get('events', [])
    
    if events:
        print(f"   [OK] Encontrados {len(events)} eventos de log\n")
        print("   Últimos eventos (más recientes primero):")
        print("   " + "-" * 66)
        for event in reversed(events[-30:]):  # Últimos 30
            timestamp = datetime.fromtimestamp(event['timestamp'] / 1000)
            message = event['message'].strip()
            # Mostrar solo líneas importantes
            if any(keyword in message.lower() for keyword in ['error', 'exception', 'traceback', 'failed', 'timeout', 'model_fn', 'predict_fn']):
                print(f"   [{timestamp.strftime('%H:%M:%S')}] {message[:200]}")
    else:
        print("   [WARN] No se encontraron logs recientes")
        print("   Esto puede ser normal si el endpoint se acaba de crear")
        
except logs_client.exceptions.ResourceNotFoundException:
    print(f"   [WARN] El log group no existe aun: {log_group_name}")
    print("   Los logs apareceran despues de la primera invocacion")
except Exception as e:
    print(f"   [ERROR] Error obteniendo logs: {e}")

# 3. Ver metricas
print("\n[3] Revisando metricas del endpoint...")
cloudwatch = boto3.client('cloudwatch', region_name=AWS_REGION)

try:
    end_time = datetime.now()
    start_time = end_time - timedelta(hours=1)
    
    metrics = cloudwatch.get_metric_statistics(
        Namespace='AWS/SageMaker',
        MetricName='ModelLatency',
        Dimensions=[
            {'Name': 'EndpointName', 'Value': ENDPOINT_NAME},
            {'Name': 'VariantName', 'Value': 'AllTraffic'}
        ],
        StartTime=start_time,
        EndTime=end_time,
        Period=60,
        Statistics=['Average', 'Maximum']
    )
    
    if metrics['Datapoints']:
        print("   [OK] Metricas encontradas:")
        for point in sorted(metrics['Datapoints'], key=lambda x: x['Timestamp']):
            print(f"   - Latencia promedio: {point.get('Average', 0):.2f}ms")
            print(f"   - Latencia maxima: {point.get('Maximum', 0):.2f}ms")
    else:
        print("   [WARN] No hay metricas aun (normal si no ha habido invocaciones)")
        
except Exception as e:
    print(f"   [WARN] No se pudieron obtener metricas: {e}")

# 4. URLs utiles
print("\n" + "=" * 70)
print("[LINKS] ENLACES UTILES")
print("=" * 70)
print(f"\nEndpoint en SageMaker:")
print(f"   https://console.aws.amazon.com/sagemaker/home?region={AWS_REGION}#/endpoints/{ENDPOINT_NAME}")

print(f"\nLogs en CloudWatch:")
print(f"   https://console.aws.amazon.com/cloudwatch/home?region={AWS_REGION}#logsV2:log-groups/log-group/%2Faws%2Fsagemaker%2FEndpoints%2F{ENDPOINT_NAME.replace('-', '%2D')}")

print(f"\nMetricas en CloudWatch:")
print(f"   https://console.aws.amazon.com/cloudwatch/home?region={AWS_REGION}#metricsV2:graph=~();namespace=AWS/SageMaker;dimensions=EndpointName,{ENDPOINT_NAME}")

print("\n" + "=" * 70)
print("[OK] Diagnostico completado")
print("=" * 70)

