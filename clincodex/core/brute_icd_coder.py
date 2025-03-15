from .brute_coder import bruteCoder
from clincodex.utils.llama_handler import llama_model,get_prompt
from clincodex.utils.gpt_handler import get_GPT_response

from langchain import PromptTemplate, LLMChain

import json

class icdbruteCoder(bruteCoder):

    def __init__(self, identifier, vectorstore, node_context_df, system_prompts,embedding_function_for_context_retrieval):
        super().__init__(identifier, vectorstore, node_context_df, system_prompts,embedding_function_for_context_retrieval)
    
    def quasi_question(self,entities):
        return '''#match icd 10 identifier for all the entities. if more than one entities are present they are sperated by || .
                ### entities ### '''+" || ".join(entities)
    
    def get_base_context_prompt(self,node_name):
        return f'''## Context for mapping ICD codes to the entity {node_name} ## '''
    
    def entity_extracter(self,question,spec_map):
        prompt_updated = self.system_prompts["DISEASE_ENTITY_EXTRACTION"] + "\n" + "Sentence : " + question
        output = get_GPT_response(instruction= prompt_updated, system_prompt=self.system_prompts["DISEASE_ENTITY_EXTRACTION"],spec_map=spec_map, temperature=spec_map["LLM_TEMPERATURE"])
        if not output:
            raise Exception(f"No response from LLM") 
        try:
            entity_dict = json.loads(output)
            return entity_dict["Diseases"]
        except:
            output = self.op_formatter(obj_str=output,spec_map=spec_map,convert_to_format="json")
            try:
                entity_dict = json.loads(output)
                return entity_dict["Diseases"]
            except:
                raise Exception(f"Unable to extract entities from {output}") 
    
    def get_llama_output(self,question,spec_map,llama_method,system_prompt,rag_flag=False,node_context_extracted=""):
        if not rag_flag:
            template = get_prompt("Question: {question}", system_prompt)
            prompt = PromptTemplate(template=template, input_variables=["question"])
            llm = llama_model(spec_map["LLAMA_MODEL_NAME"], spec_map["LLAMA_MODEL_BRANCH"], spec_map["LLM_CACHE_DIR"], stream=True, method=llama_method) 
            llm_chain = LLMChain(prompt=prompt, llm=llm)
            output = llm_chain.run(question=question)
        else:
            template = get_prompt(f"Context:\n\n{node_context_extracted} \n\nQuestion: {question}", system_prompt)
            prompt = PromptTemplate(template=template, input_variables=["context", "question"])
            llm = llama_model(spec_map["LLAMA_MODEL_NAME"], spec_map["LLAMA_MODEL_BRANCH"], spec_map["LLM_CACHE_DIR"], stream=True, method=llama_method) 
            llm_chain = LLMChain(prompt=prompt, llm=llm)
            output = llm_chain.run(context=node_context_extracted, question=question)

        return output

    def get_gpt_output(self,question,spec_map,system_prompt,rag_flag=False,node_context_extracted=""):
        if not rag_flag:
            enriched_prompt = "Question: " + question
            output = get_GPT_response(instruction= enriched_prompt, system_prompt=system_prompt,spec_map=spec_map, temperature=spec_map["LLM_TEMPERATURE"])
        else:
            enriched_prompt = "Context: "+ node_context_extracted + "\n" + "Question: " + question
            output = get_GPT_response(instruction= enriched_prompt, system_prompt=system_prompt,spec_map=spec_map, temperature=spec_map["LLM_TEMPERATURE"])
        return output