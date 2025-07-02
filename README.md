# Master-thesis
Repository for my Language &amp; AI master thesis

Uploaded files:  
• create_dataset  
&nbsp;&nbsp;&nbsp;&nbsp;- extracts data from the gold files to create SAGE-AMR  
• process_data.py  
&nbsp;&nbsp;&nbsp;&nbsp;- includes functions to:  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;merge speech and gesture AMR  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;format the AMR with indentations  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;create the dataset  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;count statistics over the dataset   
• utils_llama.py  
&nbsp;&nbsp;&nbsp;&nbsp;• includes functions to:  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;clean data for prompting  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;split data into test and train set  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;write prompt  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;generate prompt to prompt Llama  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;prompt for the error analysis  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;count statistics for the error analysis  
• Llama_thesis  
&nbsp;&nbsp;&nbsp;&nbsp;• creates test and train set  
&nbsp;&nbsp;&nbsp;&nbsp;• prompting Llama  
• error_analysis  
&nbsp;&nbsp;&nbsp;&nbsp;• classify and count error types  
&nbsp;&nbsp;&nbsp;&nbsp;• prompt Llama with the refined prompt  
&nbsp;&nbsp;&nbsp;&nbsp;• count error types in the refined prompt  
• evaluation  
&nbsp;&nbsp;&nbsp;&nbsp;• calculate BLEU, METEOR and BERTScore over the whole dataset, as well as for all separate conditions  
&nbsp;&nbsp;&nbsp;&nbsp;• paired bootstrap resampling to check whether results are significant and not due to chance  
&nbsp;&nbsp;&nbsp;&nbsp;• calculate Smatch score over the generated speechh AMRs  

In order to run your Llama model you need to create a virtual environment and install:  
llama-cpp-python  
openai  
sse_starlette  
starlette_context  
pydantic_settings  
"fastapi[all]"  
jupyter  
  
You also need to download the Llama model you want to use. I used Meta-Llama-3.1-8B-Instruct.Q4_K_M.gguf which can be found on Huggingface

