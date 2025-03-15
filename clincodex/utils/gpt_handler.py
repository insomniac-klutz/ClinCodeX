import sys
import time
import os

from tenacity import retry, stop_after_attempt, wait_random_exponential

from openai import AzureOpenAI

from clincodex.utils.base_handler import set_mem_cache_for_gpt

# Setting Cache Dir
memory=set_mem_cache_for_gpt("cachegpt") 

#@memory.cache
def get_GPT_response(instruction, system_prompt,spec_map=None, temperature=0):
    return fetch_GPT_response(instruction, system_prompt,spec_map, temperature)

#@retry(wait=wait_random_exponential(min=10, max=30), stop=stop_after_attempt(5))
def fetch_GPT_response(instruction, system_prompt,spec_map=None, temperature=0):

    if spec_map:
        OPENAI_API_KEY =spec_map["OPENAI_API_KEY"]
        OPENAI_RESOURCE_ENDPOINT = spec_map["OPENAI_RESOURCE_ENDPOINT"]
        OPEN_AI_ENGINE = spec_map["OPEN_AI_ENGINE"]
        OPEN_AI_API_VERSION = spec_map["OPEN_AI_API_VERSION"]
        OPEN_AI_AZURE_DEPLOYMENT_ID=os.environ["OPEN_AI_AZURE_DEPLOYMENT_ID"]
    else:
        OPENAI_API_KEY =os.environ["OPENAI_API_KEY"]
        OPENAI_RESOURCE_ENDPOINT = os.environ["OPENAI_RESOURCE_ENDPOINT"]
        OPEN_AI_API_VERSION = os.environ["OPENAI_API_VERSION"]
        OPEN_AI_ENGINE = os.environ["OPENAI_ENGINE"]
        OPEN_AI_AZURE_DEPLOYMENT_ID=os.environ["OPENAI_AZURE_DEPLOYMENT_ID"]

        # Setup OPENAPI

        openai_client = AzureOpenAI(
                api_key=OPENAI_API_KEY,
                azure_endpoint=OPENAI_RESOURCE_ENDPOINT,
                api_version=OPEN_AI_API_VERSION,
                azure_deployment=OPEN_AI_AZURE_DEPLOYMENT_ID)
    
    # openai_client = AzureOpenAI(
    #     api_key=OPENAI_API_KEY,
    #     base_url=OPENAI_RESOURCE_ENDPOINT,
    #     api_version=OPEN_AI_API_VERSION,
    # )

    response = openai_client.chat.completions.create(
        temperature=temperature,
        model=OPEN_AI_ENGINE,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": instruction}
        ]
    )

    return response.choices[0].message.content,cost_calc(dict(response.usage))

def stream_out(output):
    CHUNK_SIZE = int(round(len(output)/50))
    SLEEP_TIME = 0.1

    if CHUNK_SIZE > 2:
        for i in range(0, len(output), CHUNK_SIZE):
            print(output[i:i+CHUNK_SIZE], end='')
            sys.stdout.flush()
            time.sleep(SLEEP_TIME)
    else:
        print(output, end='')
        
    print("\n")

def cost_calc(usage_dict):
    cost_dict = {
                    "input_tokens":usage_dict["prompt_tokens"],
                    "output_tokens":usage_dict["completion_tokens"]
                }

    base  = 1000000 # 1 M Tokens

    ip_cost = 0.15
    op_cost = 0.6

    cost_dict["cost"] = (cost_dict["input_tokens"]/base) * ip_cost + (cost_dict["output_tokens"]/base) * op_cost

    return cost_dict
