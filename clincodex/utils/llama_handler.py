import torch

from langchain import HuggingFacePipeline
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM, TextStreamer

B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"

def get_prompt(instruction, new_system_prompt):
    system_prompt = B_SYS + new_system_prompt + E_SYS
    prompt_template =  B_INST + system_prompt + instruction + E_INST
    return prompt_template

def llama_model(model_name, branch_name, cache_dir, temperature=0, top_p=1, max_new_tokens=512, stream=False, method='method-1'):
    if method == 'method-1':
        tokenizer = AutoTokenizer.from_pretrained(model_name,
                                                 revision=branch_name,
                                                 cache_dir=cache_dir)
        model = AutoModelForCausalLM.from_pretrained(model_name,                                             
                                            device_map='auto',
                                            torch_dtype=torch.float16,
                                            revision=branch_name,
                                            cache_dir=cache_dir
                                            )
    elif method == 'method-2':
        import transformers
        tokenizer = transformers.LlamaTokenizer.from_pretrained(model_name, 
                                                                revision=branch_name, 
                                                                cache_dir=cache_dir, 
                                                                legacy=False)
        model = transformers.LlamaForCausalLM.from_pretrained(model_name, 
                                                              device_map='auto', 
                                                              torch_dtype=torch.float16, 
                                                              revision=branch_name, 
                                                              cache_dir=cache_dir)        
    if not stream:
        pipe = pipeline("text-generation",
                    model = model,
                    tokenizer = tokenizer,
                    torch_dtype = torch.bfloat16,
                    device_map = "auto",
                    max_new_tokens = max_new_tokens,
                    do_sample = True
                    )
    else:
        streamer = TextStreamer(tokenizer)
        pipe = pipeline("text-generation",
                    model = model,
                    tokenizer = tokenizer,
                    torch_dtype = torch.bfloat16,
                    device_map = "auto",
                    max_new_tokens = max_new_tokens,
                    do_sample = True,
                    streamer=streamer
                    )        
    llm = HuggingFacePipeline(pipeline = pipe,
                              model_kwargs = {"temperature":temperature, "top_p":top_p})
    return llm