import boto3

# 创建 SageMaker 客户端
sm_client = boto3.client('sagemaker')


def list_sm_endpoints():# 获取当前 AWS 账户下的所有 SageMaker 端点
    endpoint_list=[]
    endpoints = sm_client.list_endpoints()
    for endpoint in endpoints['Endpoints']:
        endpoint_name = endpoint['EndpointName']
        endpoint_list.append(endpoint_name)
        endpoint_status = endpoint['EndpointStatus']
        #print(f"Endpoint Name: {endpoint_name}, Status: {endpoint_status}")
    return endpoint_list




