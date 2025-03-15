# Streamlit app
import os
import io
import sys
import time

import streamlit as st
from contextlib import contextmanager
from io import StringIO
from streamlit.report_thread import REPORT_CONTEXT_ATTR_NAME
from threading import current_thread
import streamlit as st
import sys

from clincodex.settings import set_env
from clincodex.settings import get_config, get_sys_prompts
from clincodex.settings import get_spec_map
from clincodex.core.icd_coder import icdCoder

st.set_page_config(layout="wide")

# Import Env Vars
set_env()

# Load Prompts and Config
system_prompts=get_sys_prompts()
config_data=get_config()

# Create Specifications Map
spec_map=get_spec_map(config_data)

data_dir_path=os.path.normpath(os.path.join(os.getcwd(),"data"))
master_file_path="icd10cm-order-April-2024.txt"

icd_coder=icdCoder(identifier="ICD 10",
                system_prompts=system_prompts,
                data_dir_path=data_dir_path,
                spec_map=spec_map,
                master_file_path=master_file_path,
                verbose=True)


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


# stdout examples

# with st_stdout("code"):
#     print("Prints as st.code()")

# with st_stdout("info"):
#     print("Prints as st.info()")

# with st_stdout("markdown"):
#     print("Prints as st.markdown()")

# with st_stdout("success"), st_stderr("error"):
#     print("You can print regular success messages")
#     print("And you can redirect errors as well at the same time", file=sys.stderr)

def main():

    st.title("ICD Coder")
    
    # User input
    question = st.text_input("Enter your Note :")
    
    if question:
        
        start_time = time.time()
        with st_stdout("code"):
            # Process the question
            output = icd_coder.code(question)
        
        end_time = time.time()

        execution_time = end_time - start_time

        minutes = int(execution_time // 60)
        seconds = int(execution_time % 60)

        execution_time_message = f"Execution time: {minutes} minutes and {seconds} seconds"
        st.subheader(execution_time_message)

        # Print final output
        st.subheader("Output:")
        st.write(output)

        
    
    if st.button("Clear Output"):
        # Reload the page to clear all messages
        st.experimental_rerun()

if __name__ == "__main__":
    main()