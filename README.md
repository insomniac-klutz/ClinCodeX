# ClinCodeX

# Pipeline

- [x] Diagnosis - [ICD10CM](https://www.nlm.nih.gov/research/umls/sourcereleasedocs/current/ICD10CM)
- [ ] Medications - [RxNorm](https://www.nlm.nih.gov/research/umls/rxnorm/index.html)
- [ ] Procedures - [CPT | HCPCS](https://www.ama-assn.org/practice-management/cpt/cpt-overview-and-code-approval)
- [ ] Lab Results - [LOINC](https://www.nlm.nih.gov/research/umls/loinc_main.html)
- [ ] One Shot Unified Interface

---

# How It Works - Diagnosis - ICD10CM

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

ICD10CM

https://github.com/user-attachments/assets/e1601471-75e8-4ef9-9970-bf98b2628da4

---

