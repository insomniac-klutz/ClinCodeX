# ClinCodeX

---

# How It Works

    1. Understanding the Text
    
      The function first cleans up the given text to remove unnecessary spaces and prepares it for processing.
    
    2. Identifying Diseases (Stage A: NER - Disease Recognition)
    
      It analyzes the text to identify medical conditions and their corresponding descriptions.
      The function ensures accuracy by making multiple attempts if needed.
      At the end of this stage, a list of potential diseases is generated along with supporting information.
    
    3. Finding Relevant Guidelines (Stage B: RAG - Guidelines Retrieval)
    
      Once diseases are identified, the function looks up medical coding guidelines to find the best match.
      These guidelines provide important rules on how the diseases should be categorized and coded.
    
    4. Assigning the Right Code (Stage C: CODE - Mapping to Medical Codes)
    
      The function uses the identified diseases and guidelines to determine the correct medical codes.
      It also explains the reasoning behind each assigned code.
      If any issues arise, the function makes additional attempts to improve accuracy.
    
    5. Final Output
    
      At the end of the process, the function provides:
        A list of medical codes linked to the diseases in the text.
        A summary of the time taken and effort used in the process.

---
# Demo

https://github.com/user-attachments/assets/e1601471-75e8-4ef9-9970-bf98b2628da4

---

