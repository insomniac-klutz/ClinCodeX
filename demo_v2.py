# Streamlit app
import os
import io
import sys
import time
import tiktoken

import streamlit as st
from contextlib import contextmanager
from io import StringIO
from streamlit.report_thread import REPORT_CONTEXT_ATTR_NAME
from threading import current_thread
import streamlit as st
import sys

from clincodex.settings import set_env
from clincodex.settings import get_sys_prompts
from clincodex.core.icd_coder_v2 import icdCoder

import logging

logging.getLogger().setLevel(logging.ERROR)

for logger_name in logging.root.manager.loggerDict:
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.ERROR)

st.set_page_config(layout="wide")

# Import Env Vars
set_env()

system_prompts=get_sys_prompts()
data_dir_path=os.path.normpath(os.path.join(os.getcwd(),"data"))

RAG_INDEX_PATH = "/Users/Sankalp.C/Desktop/Neo/clincodex/clincodex/builder/.ragatouille/colbert/indexes/icd_guide_24_section_1_chapter_c"

@contextmanager
def st_redirect(src, dst):
    placeholder = st.empty()
    output_func = getattr(placeholder, dst)

    with StringIO() as buffer:
        old_write = src.write

        def new_write(b):
            if getattr(current_thread(), REPORT_CONTEXT_ATTR_NAME, None):
                buffer.write(b)
                output_func(buffer.getvalue())
            else:
                old_write(b)

        try:
            src.write = new_write
            yield
        finally:
            src.write = old_write


@contextmanager
def st_stdout(dst):
    with st_redirect(sys.stdout, dst):
        yield


@contextmanager
def st_stderr(dst):
    with st_redirect(sys.stderr, dst):
        yield

icd_coder = icdCoder(rag_index_path=RAG_INDEX_PATH)

def get_token_length(text, model="gpt-4o"):
    encoding = tiktoken.encoding_for_model(model)
    tokens = encoding.encode(text)
    return len(tokens)

def main():

    st.title("clincodex")
    
    # User input
    question = st.text_input("Enter Text Here:")

    #question = '''The patient is a 66-year-old female who presents to the clinic today for a five-month recheck on her                 adult onset diabetes mellitus along with nerve damage,                 as well as                 hypertension.                 While here she had a couple of other issues as well.                  She stated that she has been having some                  right shoulder pain.                 She denies any injury but certain range of motion does cause it to hurt.                  No weakness, numbness or tingling.                 As far as her                  diabetes                 she states that she only checks her blood sugars in the morning and those have all been ranging less than 100.                  She has not been checking any two hours after meals.                  Since childhood she has also been suffering from mild ashtama that is persistent                 Her                 blood pressures                 when she does check them have been running normal as well but she does not have any record of these present with her.                  No other issues or concerns.                 One other associated problem was edema that was of macular type and present in both left and right eyes.                 Upon review of her chart it did show that she had a                  benign breast biopsy for left breast done back on 06/11/04 and was told to have a repeat                  mammogram in six months but she has never had that done so she is needing to have this done as well.                 The patient has been also suffering from mild ckd disease.                 She had successful treatment for macular edema two years ago.'''
    
    if question:
        
        token_count = get_token_length(question)
        st.write(f"Estimated Tokens in Text : {token_count}")
        
        with st_stdout("code"):
            output,metrics,exec_time=icd_coder.code(question)

        minutes = int(exec_time // 60)
        seconds = int(exec_time % 60)

        execution_time_message = f"Execution time: {minutes} minutes and {seconds} seconds"
        st.subheader(execution_time_message)

        st.subheader("Output:")
        st.write(output)
        st.subheader("Metrics:")
        metrics['cost'] = f"$ {round(metrics['cost'], 2)}"
        metrics['cost_if_deepseek_v3'] = f"$ {round(metrics['cost_if_deepseek_v3'], 2)}"
        st.write(metrics)
    
    # if st.button("Clear Output"):
    #     # Reload the page to clear all messages
    #     st.experimental_rerun()

if __name__ == "__main__":
    main()