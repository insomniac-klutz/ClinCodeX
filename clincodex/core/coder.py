import numpy as np
import json
import sys
import time
from collections import defaultdict
import streamlit as st
import threading

import threading

lock = threading.Lock()

from tenacity import retry, stop_after_attempt,wait_random_exponential

from clincodex.utils.gpt_handler import stream_out
from clincodex.utils.gpt_handler import get_GPT_response

class Coder:
    def __init__(self, identifier,system_prompts,data_dir_path,spec_map,verbose=True):
        self.identifier = identifier
        self.system_prompts=system_prompts
        self.data_dir_path=data_dir_path
        self.spec_map=spec_map
        self.verbose=verbose

    @property
    def get_id(self):
        return self.identifier
    
    def op_formatter(self,obj_str,convert_to_format="json"):
        instruct_prompt= f'''Convert the {obj_str} to type {convert_to_format} . 
                            The output should be programmably loadable as {convert_to_format} in python as is. 
                            Do not add any suffix or prefix in the output''' 

        return get_GPT_response(instruction= instruct_prompt, system_prompt=self.system_prompts["OUTPUT_VALIDATION_CORRECTION"],spec_map=self.spec_map, temperature=self.spec_map["LLM_TEMPERATURE"])
    
    def stream_out(self,output):
        with lock:
            try:
                CHUNK_SIZE = int(round(len(output)/50))
                SLEEP_TIME = 0.1
                for i in range(0, len(output), CHUNK_SIZE):
                    print(output[i:i+CHUNK_SIZE], end='')
                    sys.stdout.flush()
                    time.sleep(SLEEP_TIME)
                print("\n")
                sys.stdout.flush()
            except:
                print(output)
    
    def entity_extracter(self,note):
        raise NotImplementedError(f"The method {self.entity_extracter.__name__} is not implemented for {self.get_id}")
    
    def process_txt(self,note):
        raise NotImplementedError(f"The method {self.entity_extracter.__name__} is not implemented for {self.get_id}")
   
    def code(self,note):
        raise NotImplementedError(f"The method {self.code.__name__} is not implemented for {self.get_id}")
