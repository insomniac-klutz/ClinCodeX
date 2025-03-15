from collections import defaultdict

import pandas as pd

def parse_icd_txt(path_to_icd_txt,create_parent_child_dict=False):
    with open(path_to_icd_txt, 'r') as file:
        lines = file.readlines()
    
    base_data_dict={}
    parent_child_dict=defaultdict(list)

    for line in lines:
        code=line[6:14].strip()
        descripts=line[77:].strip()

        base_data_dict[code]=descripts
    
    if create_parent_child_dict:
        items = base_data_dict.items()

        code_len=len(items)

        for i,k1 in enumerate(base_data_dict):
            parent_child_dict[(k1,base_data_dict[k1])] = [[k2, v2] for k2, v2 in items if k1 in k2]
            if i%1000==0:
                print(f"Done {i} of {code_len}")
    
    return base_data_dict,parent_child_dict

def build_node_context_df(parent_child_dict):

    df=pd.DataFrame(columns=["node_name","node_context"])

    code_len=len(parent_child_dict.items())

    for i,(k,v) in enumerate(parent_child_dict.items()):
        node_name=k[1]
        node_context=" . ".join(f"ICD 10 code for {item[1]} is {item[0]}" for item in v)

        entity_feat={"node_name" : node_name, "node_context": node_context}
    
        df.loc[len(df)] = entity_feat
    
        if i%1000==0:
            print(f"Done {i} of {code_len}")
    
    return df