# Master-thesis
Repository for my Language &amp; AI master thesis

Uploaded files:  
• create_dataset  
&nbsp;&nbsp;- extracts data from the gold files to create SAGE-AMR  
• process_data.py  
	- includes functions to:  
		merge speech and gesture AMR  
		format the AMR with indentations  
		create the dataset  
		count statistics over the dataset   
• utils_llama.py  
	• includes functions to:  
		clean data for prompting  
		split data into test and train set  
		write prompt  
		generate prompt to prompt Llama  
		prompt for the error analysis  
		count statistics for the error analysis  
• Llama_thesis  
	• creates test and train set  
	• prompting Llama  
• error_analysis  
	• classify and count error types  
	• prompt Llama with the refined prompt  
	• count error types in the refined prompt  
• evaluation  
	• calculate BLEU, METEOR and BERTScore over the whole dataset, as well as for all separate conditions  
	• paired bootstrap resampling to check whether results are significant and not due to chance  
	• calculate Smatch score over the generated speechh AMRs  

In order to run your Llama model you need to create a virtual environment and install:  
llama-cpp-python  
openai  
sse_starlette  
starlette_context  
pydantic_settings  
"fastapi[all]"  
jupyter  
  
You also need to download the Llama model you want to use. I used Meta-Llama-3.1-8B-Instruct.Q4_K_M.gguf which can be found on Huggingface

