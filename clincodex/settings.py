import os
import yaml
from dotenv import find_dotenv,load_dotenv
from collections import defaultdict

class KeyNameDefaultDict(defaultdict):
        def __missing__(self, key):
            self[key] = f'''{key} not present in specification map. 
                            Make sure that it is present in config map 
                            and add a reference for
                            it in clincodex.settings '''
            return self[key]

def set_env():
    env_file = find_dotenv()
    if env_file:
        load_dotenv(env_file)
    else:
        raise Exception("No valid .env file found. See env_template at project root for required env vars")

def get_config():
    with open('config.yaml', 'r') as f:
        config_data = yaml.safe_load(f)
        
    if 'GPT_CONFIG_FILE' in config_data:
        config_data['GPT_CONFIG_FILE'] = config_data['GPT_CONFIG_FILE'].replace('$HOME', os.getcwd())
    
    return config_data

def convert_to_valid_path(path):
     return os.path.normpath(os.path.join(os.getcwd(),path))

def get_sys_prompts():
    with open('system_prompts_v2.yaml', 'r') as f:
            system_prompts = yaml.safe_load(f)
    
    return system_prompts

def get_spec_map(config_data):

    spec_map=KeyNameDefaultDict()

    #Vector DB attbs

    spec_map["CHUNK_SIZE"] = int(config_data["VECTOR_DB_CHUNK_SIZE"])
    spec_map["CHUNK_OVERLAP"] = int(config_data["VECTOR_DB_CHUNK_OVERLAP"])
    spec_map["BATCH_SIZE"] = int(config_data["VECTOR_DB_BATCH_SIZE"])

    #Data locs

    spec_map["DATA_PATHS"] = {
                                }
    
    spec_map["VECTOR_DB_PATHS"] = {
                                    "spoke_vectors":convert_to_valid_path(config_data["SPOKE_VECTOR_DB_PATH"]),
                                    "icd_vectors":convert_to_valid_path(config_data["ICD_10_CM_VECTOR_DB_PATH"])
                                    }

    spec_map["NODE_CONTEXT_PATHS"] = {
                                        "icd_context_map":convert_to_valid_path(config_data["ICD_NODE_CONTEXT_PATH"]),
                                        "spoke_context_map":convert_to_valid_path(config_data["SPOKE_NODE_CONTEXT_PATH"])
                                        }

    
    #Sent Transformers ids

    spec_map["SENTENCE_EMBEDDING_MODEL"] = config_data["VECTOR_DB_SENTENCE_EMBEDDING_MODEL"]
    spec_map["SENTENCE_EMBEDDING_MODEL_FOR_NODE_RETRIEVAL"] = config_data["SENTENCE_EMBEDDING_MODEL_FOR_NODE_RETRIEVAL"]
    spec_map["SENTENCE_EMBEDDING_MODEL_FOR_CONTEXT_RETRIEVAL"] = config_data["SENTENCE_EMBEDDING_MODEL_FOR_CONTEXT_RETRIEVAL"]

    # Context Retrival vars

    spec_map["CONTEXT_VOLUME"] = int(config_data["CONTEXT_VOLUME"])
    spec_map["QUESTION_VS_CONTEXT_SIMILARITY_PERCENTILE_THRESHOLD"] = float(config_data["QUESTION_VS_CONTEXT_SIMILARITY_PERCENTILE_THRESHOLD"])
    spec_map["QUESTION_VS_CONTEXT_MINIMUM_SIMILARITY"] = float(config_data["QUESTION_VS_CONTEXT_MINIMUM_SIMILARITY"])

    # GPT vars

    spec_map["LLM_TEMPERATURE"] = config_data["LLM_TEMPERATURE"]

    spec_map["OPENAI_API_KEY"] = os.environ["OPENAI_API_KEY"]
    spec_map["OPENAI_RESOURCE_ENDPOINT"] = os.environ.get("OPENAI_RESOURCE_ENDPOINT")
    spec_map["OPEN_AI_ENGINE"] = os.environ.get("OPEN_AI_ENGINE")
    spec_map["OPEN_AI_API_VERSION"] = os.environ.get("OPEN_AI_API_VERSION")
    spec_map["OPENAI_PROMPT_RATE"] = float(os.environ.get("OPENAI_PROMPT_RATE"))
    spec_map["OPENAI_COMPLETION_RATE"] = float(os.environ.get("OPENAI_COMPLETION_RATE"))

    # To me modified after support of other types of models is introduced
    
    spec_map["CHAT_MODEL_ID"] = spec_map["OPEN_AI_ENGINE"]
    spec_map["LLM_IDENTIFIER"] = config_data["LLM_IDENTIFIER"]

    return spec_map
