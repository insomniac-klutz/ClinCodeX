import os
import re
import pickle
import json
import copy
from collections import defaultdict
import time

from .coder import Coder
from clincodex.utils.gpt_handler import get_GPT_response

from clincodex.utils.icd_handler import parse_icd_txt

import threading
from streamlit.report_thread import add_report_ctx
class icdCoder(Coder):

    def __init__(self, identifier, system_prompts,data_dir_path,spec_map,master_file_path,verbose):
        super().__init__(identifier,system_prompts,data_dir_path,spec_map,verbose)

        self.code_to_entity_dict,_=parse_icd_txt(os.path.normpath(os.path.join(data_dir_path,master_file_path)))

        self.code_to_entity_dict= {k:v.replace("(",",").replace(")",",") for k,v in self.code_to_entity_dict.items()}
        self.entity_to_code_dict= {v:k for k,v in self.code_to_entity_dict.items()}

    @property
    def get_id(self):
        return self.identifier
    
    def process_txt(self,note):
        return [re.sub(r'\s+', ' ', x).replace("(",",").replace(")",",").replace("'s","") for x in note.replace("\n"," ").split(".") if x]

    def entity_extracter(self,note_sent_piece,level_zero_context):

        instruction=f'''
                <<Text List >>
                {note_sent_piece}
        
                << Disease List >>
                {level_zero_context}
                
                '''
        system =self.system_prompts["DISEASE_ENTITY_EXTRACTION"]
         
        output = get_GPT_response(instruction= instruction, system_prompt=system,spec_map=self.spec_map, temperature=self.spec_map["LLM_TEMPERATURE"])
        
        if not output:
            raise Exception(f"No response from LLM") 
        try:
            entity_dict = json.loads(output)
            return entity_dict
        except:
            output = self.op_formatter(obj_str=output,convert_to_format="json")
            try:
                return json.loads(output)
            except:
                raise Exception(f"Unable to extract entities from {output}") 
    
    def load_lvl_zero_aug(self):
        with open(os.path.normpath(os.path.join(self.data_dir_path,'disease.pkl')), 'rb') as f:
            disease_list = pickle.load(f)
        
        # context_str=""
        # for i,disease in enumerate(disease_list):
        #     context_str+=f"{i+1}. {disease}\n" 
        
        return disease_list

    def get_level_zero_codes(self,level_zero_entity_dict):
        
        code_list=[]

        for key in level_zero_entity_dict.keys():
            code=key.split("||")[1].strip()
            try:
                _=self.code_to_entity_dict[code]
                code_list.append(code)
            except:
                if "." in code \
                    and len(code.split(".")[0])==3 \
                        and code.split(".")[0] in self.code_to_entity_dict.keys():

                    code_list.append(code.split(".")[0])

                    thread=threading.Thread(target=self.stream_out, args=(f"Attention User !!! - I found an invalid root code {code}",))
                    add_report_ctx(thread)
                    thread.start()

                    thread=threading.Thread(target=self.stream_out, args=(f"Reverting it to its head - {code.split('.')[0]}",))
                    add_report_ctx(thread)
                    thread.start()

                    # self.stream_out(f"Attension User !!! - I found an invalid root code {code}")
                    # self.stream_out(f"Reverting it to its head - {code.split('.')[0]}")
                else:
                    thread=threading.Thread(target=self.stream_out, args=(f"Attension User !!! - I found an invalid root code {code}",))
                    add_report_ctx(thread)
                    thread.start()

                    thread=threading.Thread(target=self.stream_out, args=(f"Unable to revert it to its head . Removing entity from consideration",))
                    add_report_ctx(thread)
                    thread.start()

                    # self.stream_out(f"Attension User !!! - I found an invalid root code {code}")
                    # self.stream_out(f"Unable to revert it to its head . Removing entity from consideration")
        return code_list
    
    def get_leveler_dict(self,node_data_dict,relevant_keys):
        data_leveler_dict=defaultdict(list)

        base_key=copy.deepcopy(relevant_keys)

        while(base_key):
            can_key=[]
            for key,_ in node_data_dict.items():
                for parent_key in base_key:
                    if (key.startswith(parent_key) and (len(key)==len(parent_key)+1))\
                        or (key.startswith(parent_key+"X") and (len(key)==len(parent_key+"X")+1)):
                        data_leveler_dict[parent_key].append(key)
                        can_key.append(key)

            base_key=can_key

        return data_leveler_dict
    
    def get_context_questions(self,parent_text,children_txt_list):

        instruction=f'''
                << Parent Node >>
                {parent_text}
        
                << Children Node(s) >>
                {children_txt_list}
                
                '''
        
        system =self.system_prompts["QNA_GENERATION_CHILD_NODES"]
         
        output = get_GPT_response(instruction= instruction, system_prompt=system,spec_map=self.spec_map, temperature=self.spec_map["LLM_TEMPERATURE"])
        
        if not output:
            raise Exception(f"No response from LLM") 
        try:
            return json.loads(output)
        except:
            output = self.op_formatter(obj_str=output,convert_to_format="json")
            try:
                return json.loads(output)
            except:
                raise Exception(f"Unable to extract entities from {output}") 

    def get_node_affirmation(self,note_sent_piece,context_questions_dict):
        instruction=f'''
                <<Text List>>
                {note_sent_piece}
                
                <<Entity Dict>>
                {context_questions_dict}

                '''
        
        system =self.system_prompts["CONTEXUAL_QNA_CHILD_NODES"]
         
        output = get_GPT_response(instruction= instruction, system_prompt=system,spec_map=self.spec_map, temperature=self.spec_map["LLM_TEMPERATURE"])
        
        if not output:
            raise Exception(f"No response from LLM") 
        try:
            return json.loads(output)
        except:
            output = self.op_formatter(obj_str=output,convert_to_format="json")
            try:
                return json.loads(output)
            except:
                raise Exception(f"Unable to extract entities from {output}") 

    def get_node_verdict(self,note_sent_piece,key,level_candidates):

        self.stream_out(f"Current level candidates --> {level_candidates}")
        parent_text=key+": "+self.code_to_entity_dict[key]
        children_txt_list = [child+": "+self.code_to_entity_dict[child] for child in level_candidates]
        positive_list=[]

        thread=threading.Thread(target=self.stream_out, args=(f"Generating context questions for this level ...",))
        add_report_ctx(thread)
        thread.start()

        #self.stream_out(f"Generating context questions for this level ...")

        context_questions_dict=self.get_context_questions(parent_text,children_txt_list)

        thread=threading.Thread(target=self.stream_out, args=(f"Performing context based QnA for this level ...",))
        add_report_ctx(thread)
        thread.start()

        #self.stream_out(f"Performing context based QnA for this level ...")

        candidate_affirmation_dict=self.get_node_affirmation(note_sent_piece,context_questions_dict)
        
        for candidate,affirmation in candidate_affirmation_dict.items():
            if affirmation=="Yes":
                positive_list.append(candidate)
        
        if self.verbose:
            for i,key in enumerate(positive_list):
                #self.stream_out(f"      {i+1}. Questions asked in affirmative for {key} : {context_questions_dict[key]}")
                thread=threading.Thread(target=self.stream_out, args=(f"      {i+1}. Questions asked in affirmative for {key} : {context_questions_dict[key]}",))
                add_report_ctx(thread)
                thread.start()

        return [x.split(":")[0] for x in positive_list]

    def get_best_code(self,node,data_leveler_dict,note_sent_piece):

        relevant_keys=[node]
        result_keys=[]

        while(relevant_keys):
            next_relevant_list=[]
            for key in relevant_keys:
                level_candidates=data_leveler_dict[key]

                positive_code_list=self.get_node_verdict(note_sent_piece,key,level_candidates)

                result_keys+=[k for k in positive_code_list if k and k!='[]' and k not in data_leveler_dict.keys()]
                next_relevant_list+=[k for k in positive_code_list if k and k != '[]' and k in data_leveler_dict.keys()]

                thread=threading.Thread(target=self.stream_out, args=(f"Positive candidates for {key} --> {positive_code_list} out of {level_candidates}",))
                add_report_ctx(thread)
                thread.start()

                thread=threading.Thread(target=self.stream_out, args=(f"Result Keys till {key} : {result_keys}",))
                add_report_ctx(thread)
                thread.start()

                thread=threading.Thread(target=self.stream_out, args=(f"Next Relevant Keys till {key} : {next_relevant_list}",))
                add_report_ctx(thread)
                thread.start()
                # self.stream_out(f"Positive candidates for {key} --> {positive_code_list} out of {level_candidates}")
                # self.stream_out(f"Result Keys till {key} : {result_keys}")
                # self.stream_out(f"Next Relevant Keys till {key} : {next_relevant_list}")
            
            relevant_keys=next_relevant_list
        
        return result_keys

    def code(self,note):
        
        note_sent_piece=self.process_txt(note)

        level_zero_context=self.load_lvl_zero_aug()

        if self.verbose:
            thread=threading.Thread(target=self.stream_out, args=(" ☛  Extracting Entity Heads",))
            add_report_ctx(thread)
            thread.start()
            #self.stream_out(" ☛  Extracting Entity Heads")
        
        level_zero_entity_dict=self.entity_extracter(note_sent_piece,level_zero_context)

        if self.verbose:
            for i,(entity,evidence) in enumerate(level_zero_entity_dict.items()):
                #self.stream_out(f"      {i+1}. '{entity}' extracted against evidence '{evidence}'")
                thread=threading.Thread(target=self.stream_out, args=(f"      {i+1}. '{entity}' extracted against evidence '{evidence}'",))
                add_report_ctx(thread)
                thread.start()
        
        if self.verbose:
            #self.stream_out(f"☛  Extracting Root ICD codes")
            thread=threading.Thread(target=self.stream_out, args=(f"☛  Extracting Root ICD codes",))
            add_report_ctx(thread)
            thread.start()

        level_zero_code_list= self.get_level_zero_codes(level_zero_entity_dict)

        if self.verbose:
            #self.stream_out(f"☛  Parent ICD codes : {level_zero_code_list}")
            thread=threading.Thread(target=self.stream_out, args=(f"☛  Parent ICD codes : {level_zero_code_list}",))
            add_report_ctx(thread)
            thread.start()

        return_code_dict=defaultdict(str)

        for i,code in enumerate(level_zero_code_list):

            if self.verbose:
                #self.stream_out(f"☛  Starting Probe for {code}:{self.code_to_entity_dict[code]}")
                thread=threading.Thread(target=self.stream_out, args=(f"☛  Starting Probe for {code}:{self.code_to_entity_dict[code]}",))
                add_report_ctx(thread)
                thread.start()

            node_data_dict={k:v for k,v in self.code_to_entity_dict.items() if k.startswith(code)}

            relevant_keys=[code]

            data_leveler_dict= self.get_leveler_dict(node_data_dict,relevant_keys)

            start_time = time.time()
            code_list=self.get_best_code(relevant_keys[0],data_leveler_dict,note_sent_piece)
            end_time = time.time()

            execution_time = end_time - start_time

            minutes = int(execution_time // 60)
            seconds = int(execution_time % 60)

            #self.stream_out(f"☛  Probe Complete in {minutes} minutes and {seconds} seconds for {code}")
            thread=threading.Thread(target=self.stream_out, args=(f"☛  Probe Complete in {minutes} minutes and {seconds} seconds for {code}",))
            add_report_ctx(thread)
            thread.start()


            if code_list:
                for res in code_list:
                    return_code_dict[res]=self.code_to_entity_dict[res]
            else:
                if code not in data_leveler_dict.keys() or not data_leveler_dict[code]:
                    code_list.append(code)
                    return_code_dict[code]=self.code_to_entity_dict[code]
                else:   
                    return_code_dict[code]=f"No Billable code found for Level Zero Code {code} : {self.code_to_entity_dict[code]}"

            if self.verbose:
                if code_list:
                    self.stream_out(f"☛  Best match leaves for {code} extracted")
                    for i,code in enumerate(code_list):
                        #self.stream_out(f"      {i+1}. '{code}' extracted with affirmed description '{self.code_to_entity_dict[code]}'")
                        thread=threading.Thread(target=self.stream_out, args=(f"      {i+1}. '{code}' extracted with affirmed description '{self.code_to_entity_dict[code]}'",))
                        add_report_ctx(thread)
                        thread.start()
                else:
                    #self.stream_out(f"      '{code}' unresolved\n")
                    thread=threading.Thread(target=self.stream_out, args=(f"      '{code}' unresolved\n",))
                    add_report_ctx(thread)
                    thread.start()

                
        
        if self.verbose:
            #self.stream_out(f"☛  Results\n")
            thread=threading.Thread(target=self.stream_out, args=(f"☛  Results\n",))
            add_report_ctx(thread)
            thread.start()

            for i,(code,descripts) in enumerate(return_code_dict.items()):
                #self.stream_out(f"      {i+1}. '{code}':'{descripts}'")
                thread=threading.Thread(target=self.stream_out, args=(f"      {i+1}. '{code}':'{descripts}'",))
                add_report_ctx(thread)
                thread.start()

        return return_code_dict

        
        

