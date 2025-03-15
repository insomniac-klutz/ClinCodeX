import pandas as pd
from collections import defaultdict
import pickle
import os
import re
import yaml

import json
from dotenv import find_dotenv,load_dotenv
from ..utils.claude_handler import claudeSonnet
from ..utils.gpt_handler import get_GPT_response
from ..utils.rag_handler import colbertRetriver

from graphviz import Digraph

import time
import pprint

class icdCoder():

    def __init__(self,rag_index_path):

        self.set_env()
        self.data_base_path=os.getcwd().split("clincodex")[0]+"/data/"

        with open(self.data_base_path+'parent_child_dict.pkl', 'rb') as f:
            self.parent_child_dict = pickle.load(f)

        with open(self.data_base_path+"disease.pkl", 'rb') as f:
            self.disease_list = pickle.load(f)

        with open(self.data_base_path+"icd10cm-order-April-2024.txt", 'r') as file:
            lines = file.readlines()
        
        self.base_data_dict={}
        #parent_child_dict=defaultdict(list)

        for line in lines:
            code=line[6:14].strip()
            descripts=line[77:].strip()

            self.base_data_dict[code]=descripts
        
        self.context_str=self.disease_list
        self.client=claudeSonnet()
        self.rag_client = colbertRetriver(rag_index_path)
        
    @staticmethod
    def set_env():
        env_file = find_dotenv()
        if env_file:
            load_dotenv(env_file)
        else:
            raise Exception("No valid .env file found. See env_template at project root for required env vars")
    
    @staticmethod
    def medgrapher(t2dm_data_list):
        graph = Digraph(format='png', node_attr={'shape': 'box', 'style': 'rounded'})

        for code, description in t2dm_data_list.items():
            if all(child[:len(code)] != code for child in t2dm_data_list if child != code):
                graph.node(code, f"{code}: {description} (billable)")#, style="filled", fillcolor="lightblue")
            else:
                graph.node(code, f"{code}: {description}")
            if len(code) > 3:
                graph.edge(code[:len(code)-1], code)
        
        dot_string = graph.source

        nodes_pattern = r'([A-Za-z0-9]+) \[label="([^"]+)"(?: [^]]*)?\]'
        edges_pattern = r'([A-Za-z0-9]+) -> ([A-Za-z0-9]+)'

        nodes = re.findall(nodes_pattern, dot_string)
        edges = re.findall(edges_pattern, dot_string)

        node_edges = defaultdict(list)

        for start, end in edges:
            node_edges[start].append(end)

        cleaned_string = []

        for node, label in nodes:
            edge_list = " | ".join(node_edges[node]) 
            if edge_list:
                cleaned_string.append(f"{label} -> {edge_list}")
            else:
                cleaned_string.append(f"{label}")

        condensed_graph = "\n".join(cleaned_string)

        return condensed_graph
    
    def code(self,txt):

        main_start_time = time.time()

        user_input = re.sub(r'\s+', ' ', txt)

        with open(os.getcwd().split("clincodex")[0]+"/system_prompts_v2.yaml", 'r') as f:
            system_prompts = yaml.safe_load(f)

        ner_system_prompt=system_prompts["DISEASE_ENTITY_EXTRACTION"]

        start_time = time.time()

        pprint.pprint("Stage A : NER - Root ICD Entity Identification")
        print("\n")

        retry = 0
        while(retry<3):
            try : 
                ner_agent_res , ner_cost_metric =self.client.claude_wrapper(ner_system_prompt,user_input=f'''
                        <<Text List >>
                        {user_input}
                
                        << Disease List >>
                        {self.context_str}
                        
                        ''')
                
                # ner_agent_res , ner_cost_metric = get_GPT_response(system_prompt=ner_system_prompt,instruction=f'''
                #                 <<Text List >>
                #                 {user_input}
                        
                #                 << Disease List >>
                #                 {self.context_str}
                                
                #                 '''
                #                 )
                
                ner_agent_res = json.loads(ner_agent_res)

                coder_agent_pack={}

                for key,value in ner_agent_res.items():
                    descript,code=key.split("||")
                    coder_agent_pack[code.strip()] = [descript,value]

                break
            except Exception as e:
                retry+=1
                if retry <3:
                    print(f"NER Agent : Unprocessable Entity with error {e} . Retry Counter ({retry})")

        end_time = time.time()

        execution_time = end_time - start_time

        minutes = int(execution_time // 60)
        seconds = int(execution_time % 60)

        print(f"NER Agent TAT: {minutes} minutes and {seconds} seconds")
        print(f"NER Agent Metrics [Sonnet]: {ner_cost_metric}\n")
        #print(f"NER Agent Metrics [4o mini]: {ner_cost_metric}")
        print(f"NER Agent Result : Generated {len(coder_agent_pack)} base candidates for probe :\n")

        for i,(k,v) in enumerate(coder_agent_pack.items()):
            print(f"\t {i+1}. Code : {k} <> Description : {v[0]} ") #<> Evidence {v[0]} 
        
        print("\n")

        pprint.pprint("Stage B : RAG - Fetch Relevant Coding Guidelines")
        print("\n")

        start_time = time.time()

        rag_guidelines,meta_content = self.rag_client.get_guidelines(coder_agent_pack)

        rag_guidelines=rag_guidelines.replace(r"\n", " ")

        end_time = time.time()

        execution_time = end_time - start_time

        minutes = int(execution_time // 60)
        seconds = int(execution_time % 60)

        print(f"RAG Agent TAT: {minutes} minutes and {seconds} seconds\n")
        print(f"RAG Agent Extracts:\n\n{meta_content}")
        
        for key,value in coder_agent_pack.items():
            t2dm_data_list={k:v for k,v in self.base_data_dict.items() if k.startswith(key)}

            value.append(self.medgrapher(t2dm_data_list))
        
        code_agent_sliced = {k:str(v[2]) for k,v in coder_agent_pack.items()}
            
        coder_system_prompt=system_prompts["DISEASE_CODING"]

        coding_guidelines = f'<< Coding Guidelines >>\n{rag_guidelines}\n'

        coder_system_prompt=coder_system_prompt.replace("<< Coding Guidelines >>",coding_guidelines)

        pprint.pprint("Stage C : CODE - Map text to Code using guidelines and ICD Sub Graphs")
        print("\n")

        start_time = time.time()

        # print(coder_system_prompt)

        retry = 0
        while(retry<3):
            try : 
                coder_agent_res, coder_cost_metric=self.client.claude_wrapper(coder_system_prompt,user_input=f'''
                                <<Text List >>
                                {user_input}
                        
                                << Disease Dictionary >>
                                {code_agent_sliced}
                                
                                ''')
                
                #print(coder_agent_res)
                
                coder_agent_reasons = " "
                # Regular expression to capture the list inside the square brackets
                pattern = r"### REFLECTION_START ###\s*([^\]]+)\s*### REFLECTION_END ###"

                # Search for the pattern
                match = re.search(pattern, coder_agent_res)

                if match:
                    coder_agent_reasons = match.group(1).strip()
                    coder_agent_reasons = re.sub(r'\n+', '\n', coder_agent_reasons)

                # Regular expression to capture the list inside the square brackets
                pattern = r"### RESULT_START ###\s*\[([^\]]+)\]\s*### RESULT_END ###"

                # Search for the pattern
                match = re.search(pattern, coder_agent_res)

                if match:
                    # Extract and clean up the list (split by commas)
                    coder_agent_res = match.group(1).strip()

                coder_agent_res = [code.strip().replace('"','') for code in coder_agent_res.split(',')]
                #coder_agent_res = json.loads(coder_agent_res)

                final_res = {}
                for code in coder_agent_res:
                    final_res[code]=self.base_data_dict[code]
                break
            except Exception as e:
                retry+=1
                if retry <3:
                    print(f"Coder Agent : Unprocessable Entity with error {e} . Retry Counter ({retry})")
        

        end_time = time.time()

        execution_time = end_time - start_time

        minutes = int(execution_time // 60)
        seconds = int(execution_time % 60)

        print(f"Coder Agent TAT: {minutes} minutes and {seconds} seconds")
        print(f"CODER Agent Metrics [Sonnet]:\n {coder_cost_metric}\n")
        print(f"CODER Agent Reasoning [Sonnet]:\n {coder_agent_reasons}\n")
        print(f"CODER Agent Result [Sonnet]:\n {coder_agent_res}")

        main_end_time = time.time()
        
        return final_res,\
            {key: ner_cost_metric.get(key, 0) + coder_cost_metric.get(key, 0) for key in set(ner_cost_metric) | set(coder_cost_metric)},\
                main_end_time - main_start_time

    
