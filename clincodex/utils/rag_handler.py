import re
import sys
import os
import io

from collections import defaultdict

from langchain_community.document_loaders import PyPDFLoader
from ragatouille import RAGPretrainedModel

# class LoggerHook(logging.Handler):
#     def emit(self, record):
#         print(f"Captured log from {record.name}: {record.getMessage()}")

# # Attach the hook
# hook = LoggerHook()
# logging.getLogger().addHandler(hook)

# def get_registered_loggers():
#     return [key for key in logging.Logger.manager.loggerDict.keys() if "torch" in key]

# print("Registered loggers:", get_registered_loggers())

class colbertRetriver:
    
    def __init__(self,index_path=None):
        
        if index_path:
            self.RAG = RAGPretrainedModel.from_index(index_path,verbose=0)
        else:
            self.RAG = RAGPretrainedModel.from_pretrained("colbert-ir/colbertv2.0",verbose=0)
            
    
    @staticmethod
    def pdf_loader(path_to_pdf):

        return PyPDFLoader(path_to_pdf).load_and_split()

    @staticmethod
    def icd_guide_cleaner(code_guide_txt):
        
        # 1) Remove repeated noise (headers, footers, disclaimers).
        patterns_to_remove = [
            r"(?i)FY\s*2024\s*ICD-10-CM\s*Official\s*Guidelines.*",  # Example header
            r"(?i)Centers\s*for\s*Medicare\s*&\s*Medicaid\s*Services.*",
            r"Page\s*\d+\s+of\s+\d+",                               # Typical footer
            r"(?i)Updated\s*\d{2}/\d{2}/\d{4}",                     # e.g., "Updated 02/01/2024"
        ]

        cleaned_text = code_guide_txt

        for pattern in patterns_to_remove:
            cleaned_text = re.sub(pattern, "", cleaned_text, flags=re.MULTILINE)

        # 2) Fix hyphenated words broken across lines (e.g., "classi-\nfication" => "classification").
        cleaned_text = re.sub(r"(\w)-\n(\w)", r"\1\2", cleaned_text)

        # 3) Convert multiple line breaks into a single line break.
        cleaned_text = re.sub(r"\n+", "\n", cleaned_text)

        # 4) Trim leading/trailing whitespace.
        cleaned_text = cleaned_text.strip()

        # 5) Merge lines into coherent paragraphs based on punctuation & length heuristics.
        lines = cleaned_text.split("\n")
        merged_lines = []
        paragraph_buffer = []

        for line in lines:
            line = line.strip()
            # Blank line => paragraph boundary
            if not line:
                if paragraph_buffer:
                    merged_lines.append(" ".join(paragraph_buffer))
                    paragraph_buffer = []
                continue

            # If line doesn't end with common sentence punctuation or is very short,
            # we assume it's a continuation of the same paragraph.
            if (len(line) < 80) or (not re.search(r"[.?!;:,]\s*$", line)):
                paragraph_buffer.append(line)
            else:
                paragraph_buffer.append(line)
                merged_lines.append(" ".join(paragraph_buffer))
                paragraph_buffer = []

        # Append any leftover text in the paragraph buffer
        if paragraph_buffer:
            merged_lines.append(" ".join(paragraph_buffer))

        # Reassemble paragraphs with double newlines
        final_cleaned = "\n\n".join(merged_lines).strip()

        return final_cleaned
    
    def extract_section_1_chapter_c(text):

        pattern = r"(?<=1\. Chapter 1: Certain Infectious and Parasitic Diseases).*?(?=Section II\. Selection of Principal Diagnosis)"

        matches = list(re.finditer(pattern, text, re.DOTALL))

        # Get the second match if it exists
        if len(matches) >= 2:
            second_match = matches[1].group().strip()  # Use .strip() to remove leading/trailing whitespace
            print("Second Match:")
            print("Chapter 1: Certain Infectious and Parasitic Diseases " + second_match)
            code_guide = "1. Chapter 1: Certain Infectious and Parasitic Diseases " + second_match
        else:
            print("Less than two matches found.")
        
        # Regular expression pattern
        pattern = r"\d+\.\sChapter\s\d+:"

        # Split the text based on the pattern
        segments = re.split(f"({pattern})", code_guide)

        # Combine the matched patterns with their corresponding text
        code_guide_chapters = ["".join(x) for x in zip(segments[1::2], segments[2::2])]

        # # Print the chapters
        # for i, chapter in enumerate(chapters, 1):
        #     print(f"Chapter {i}:\n{chapter}\n")

        return code_guide_chapters
    
    def index(self,collection,document_ids,document_metadatas,index_name): 
        self.RAG.index(
            collection=collection,
            index_name=index_name,
            max_document_length=512,
            split_documents=False,
            document_ids=document_ids,
            document_metadatas=document_metadatas,
            overwrite_index=False
        )
    
    def serach(self,query,k=3): 

        res = self.RAG.search(query=query, k=k)

        return res
    
    def get_guidelines(self,coder_action_pack): 
        
        query_list = [f"{key} - {value[0]}" for key, value in coder_action_pack.items()]

        doc_merged_content = ""
        meta_merged_content = ""
        escape = "\n\t\t"

        contents = defaultdict(list)
        rag_compress = defaultdict(list)

        for query in query_list:
            rag_search_res = self.RAG.search(query=query)

            for item in rag_search_res[:1]:
                key_index = item['document_metadata']["index"]
                contents[key_index] = [item['document_metadata']["description"],item['content']]
                rag_compress[key_index].append(query)

        for i,(k,v) in enumerate(rag_compress.items()):
            doc_merged_content += f'{i+1}. Code Guidelines for queryset [{",".join(v)}]\n'+contents[k][1]+"\n"
            meta_merged_content += f"{i+1}. Extracted Guideline {str(k) + ' - ' + contents[k][0]} for queryset:{escape}- {f'{escape}- '.join(v)}\n\n"

            # for item in rag_search_res[:1]:
            #     contents.append(item['content'])
            #     document_metadatas.append(str(item['document_metadata']["index"])+" - "+str(item['document_metadata']["description"]))
            
            # doc_merged_content += f'Code Guidelines for {query}\n'+'\n'.join(contents)+"\n"
            # meta_merged_content += f"\tExtracted Guidelines for {query}:\n" + '\n'.join(f"\t\t- {item}" for item in document_metadatas) + "\n"

            
        return doc_merged_content,meta_merged_content