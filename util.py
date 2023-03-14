import boto3

sm_client = boto3.client('sagemaker')


def list_sm_endpoints():
    endpoint_list=[]
    endpoints = sm_client.list_endpoints()
    for endpoint in endpoints['Endpoints']:
        endpoint_name = endpoint['EndpointName']
        endpoint_list.append(endpoint_name)
        endpoint_status = endpoint['EndpointStatus']
        #print(f"Endpoint Name: {endpoint_name}, Status: {endpoint_status}")
    return endpoint_list

def get_bucket_and_key(s3uri):
    pos = s3uri.find('/', 5)
    bucket = s3uri[5 : pos]
    key = s3uri[pos + 1 : ]
    return bucket, key


