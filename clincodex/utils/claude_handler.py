import boto3
import os
import json

import yaml

import logging

boto3_logger = logging.getLogger('boto3')
botocore_logger = logging.getLogger('botocore')

boto3_logger.setLevel(logging.ERROR)
botocore_logger.setLevel(logging.ERROR)

class claudeSonnet:
    _client = None

    def get_client(self):
        if claudeSonnet._client is not None:
            return claudeSonnet._client
        else:
            ANTHROPIC_RESOURCE_ENDPOINT =os.environ["ANTHROPIC_RESOURCE_ENDPOINT"]
            ANTRHOPIC_ACCESS_KEY_ID = os.environ["ANTRHOPIC_ACCESS_KEY_ID"]
            ANTRHOPIC_SECRET_ACCESS_KEY = os.environ["ANTRHOPIC_SECRET_ACCESS_KEY"]
            ANTHROPIC_REGION=os.environ["ANTHROPIC_REGION"]

            claudeSonnet._client = boto3.client('bedrock-runtime',
                                                endpoint_url=ANTHROPIC_RESOURCE_ENDPOINT,
                                                aws_access_key_id=ANTRHOPIC_ACCESS_KEY_ID,
                                                aws_secret_access_key=ANTRHOPIC_SECRET_ACCESS_KEY,
                                                region_name=ANTHROPIC_REGION
                                            )
    
        return claudeSonnet._client
    
    @staticmethod
    def load_yaml_from_string(yaml_string):
        try:
            # Load the YAML string into a Python dictionary
            data = yaml.safe_load(yaml_string)
            return data
        except yaml.YAMLError as e:
            print(f"Error loading YAML: {e}")
            return None

    def generate_message(self,bedrock_runtime, model_id, system_prompt, messages, max_tokens, temperature=1):
        body=json.dumps(
            {
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": max_tokens,
                "system": system_prompt,
                "temperature": temperature,
                "messages": messages
            }  
        )  
        response = bedrock_runtime.invoke_model(body=body, modelId=model_id)
        response_body = json.loads(response.get('body').read())
    
        return response_body
    
    def claude_wrapper(self,system_prompt, user_input, temperature=0.5):

        model_id = os.environ["ANTHROPIC_MODEL_ID"]
        max_tokens = 4000

        user_message =  {"role": "user", "content": user_input}
        messages = [user_message]

        response = self.generate_message(self.get_client(), model_id, system_prompt, messages, max_tokens, temperature=temperature) 
        # print(response['usage'])
        out = response['content'][0]['text']

        return out,self.calc_cost(response['usage'])
    
    @staticmethod
    def calc_cost(usage_dict):
        cost_dict = {k:v for k,v in usage_dict.items()}

        base  = 1000000 # 1 M Tokens

        ip_cost = 3
        op_cost = 15

        cost_dict["cost"] = (cost_dict["input_tokens"]/base) * ip_cost + (cost_dict["output_tokens"]/base) * op_cost

        ip_cost = 0.27
        op_cost = 1.10

        cost_dict["cost_if_deepseek_v3"] = (cost_dict["input_tokens"]/base) * ip_cost + (cost_dict["output_tokens"]/base) * op_cost

        return cost_dict