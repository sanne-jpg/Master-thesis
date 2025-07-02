import pandas as pd
import re

def clean_data_for_prompting(all_merged_data):
    """
    Remove time stamps and add num_gesture_amrs to each item for clarity.
    Args:
        all_merged_data: list containing all dataset information
    Returns:
        cleaned_data: list containing dataset without time stamps
        df_dataset: dataframe of the dataset
        df_by_gesture_count: dataframe of the dataset also containing num_gesture_amrs
    """
    cleaned_data = []                                 ## Remove time stamps for prompting
    for dict in all_merged_data:
        cleaned_dict = {}
        for k, v in dict.items():
            if k not in ['begin_time', 'end_time']:
                cleaned_dict[k] = v
        cleaned_data.append(cleaned_dict)
    
    for item in cleaned_data:
        item["num_gesture_amrs"] = len(item.get("gesture_amrs", []))
        
    df_dataset = pd.DataFrame(cleaned_data, columns=['file', 'sentence', 'speech_amr', 'gesture_labels', 'gesture_amrs', 'num_gesture_amrs'])
    df_by_gesture_count = {
        1: df_dataset[df_dataset['num_gesture_amrs'] == 1],
        2: df_dataset[df_dataset['num_gesture_amrs'] == 2],
        3: df_dataset[df_dataset['num_gesture_amrs'] == 3],
        4: df_dataset[df_dataset['num_gesture_amrs'] == 4],
    }

    return cleaned_data, df_dataset, df_by_gesture_count

def pretty_amr(raw_amr):
    """
    Format the input as pretty, readable AMR. 
    Copied from process_data.py
    Args:
        raw_amr: list containing linear AMR
    Returns:
        output: string containing AMR graph with indentation
    """
    output = ""
    indent = 0                                                    ## Start with the first indentation level
    extra_indent = 0                                              ## For roles that follow reference values
    i = 0
    length = len(raw_amr)

    if isinstance(raw_amr, list):
        for item in raw_amr:
            reused_var = count_op1_op2_reuse(item)
    else:
        reused_var = count_op1_op2_reuse(raw_amr)
    op1_indent_level = None
    after_op2_ref = False
   
    while i < length:
        ## Handle opening parenthesis '('
        if raw_amr[i] == '(':                                     
            if i > 0 and raw_amr[i-1] not in ['\n', ' ', '\t']:   
                output += "\n"
            output += "("                                         
            indent += 1                                           ## Increase indentation for nested structures
            i += 1

        ## Handle closing parenthesis ')'
        elif raw_amr[i] == ')':                                   
            indent -= 1                                           ## Decrease indentation when closing a block
            output += ")" 
            i += 1

        ## Handle roles (e.g., :ARG0, :op1)
        elif raw_amr[i] == ':':
            role = ""
            i += 1
            while i < length and raw_amr[i] not in [' ', '\t', '(', ')']:     ## Collect the role name after : and before (
                role += raw_amr[i]
                i += 1

            while i < length and raw_amr[i] in [' ', '\t']:                   ## Skip whitespaces
                i += 1

            # Peek next token
            temp_i = i
            var_peek = ""
            while temp_i < length and raw_amr[temp_i] not in [' ', '\t', ')', '\n', '(']:   ## Without advancing i, check the token after
                var_peek += raw_amr[temp_i]                                                 ## the role. if :op2 s, var_peek = s
                temp_i += 1

            # Store indent level for :op1
            if role == "op1" and reused_var and raw_amr[temp_i:temp_i+len(reused_var)] == reused_var:   ## If op1, and reused variable is next,
                op1_indent_level = indent                                                               ## check indent level of op1

            # Handle reused :op2 s
            if role == "op2" and reused_var and var_peek == reused_var:                               ## If op2 and next token = reused_var,
                actual_indent = op1_indent_level if op1_indent_level is not None else max(0, indent)  ## we indent op2 the same as op1
                output += "\n" + "\t" * actual_indent + f":{role} {reused_var}"                       ## Write down the full role
                i = temp_i                                                                  ## Increase i
                after_op2_ref = True                                                        ## Mark the re-using variable
            else:
                extra = 1 if after_op2_ref else 0                                    ## Indent the following roles if after_op2_ref is true
                output += "\n" + "\t" * (indent + extra) + f":{role} "
        else:
            ## Handle tokens like (x / signaler) or (g / gesture-unit)
            token = ""
            while i < length and raw_amr[i] not in [':', ' ', '\t', '(', ')']:
                token += raw_amr[i]
                i += 1
            output += token

        # Skip spaces or tabs
        if i < length and raw_amr[i] in [' ', '\t']:
            i += 1

    return output

def count_op1_op2_reuse(amr_text):
    """
    Function that checks if an AMR reuses the same variable from :op1 in :op2.
    Copied from process_data.py
    Args:
        amr_text: list containing linear AMR
    Returns:
        op1_var: if a variable is reused the abbreviated variable name is returned
        None: if there is no match return None
    """
    op1_match = re.search(r':op1\s*\(\s*([a-zA-Z0-9_-]+)\s*/', amr_text)             ## Look for the variable introduced in :op1 (...)
    
    if op1_match:                                                                    ## If we found it, extract the variable. e.g. s or i
        op1_var = op1_match.group(1)
        op2_reuses = re.findall(r':op2\s+' + re.escape(op1_var) + r'\b', amr_text)   ## Look for reuses of the same variable in :op2
        return op1_var
        
    return None

def split_data(df, test_size=0.3, seed_value=12):
    """
    Split data into an example set (70%) and a test set (30%) for each video.
    Args:
        df: dataframe containing the entire dataset
        test_size: specify how much of the total the test set should comprise
        seed_value: set a random seed to ensure the data is split the same way every time
    Returns:
        example_set: train set containing examples
        test_set: test set
    """
    # To store the example and test sets
    example_set = []
    test_set = []
    
    # Group by the file
    grouped = df.groupby('file')

    # Iterate through each group 
    for video, group in grouped:
        # Shuffle the rows within the group for randomness
        group = group.sample(frac=1, random_state=seed_value).reset_index(drop=True)

        # Split the data (70% for example set, 30% for test set)
        n_test = int(len(group) * test_size)
        
        # The first 70% are for the example set
        example_set.append(group.iloc[:-n_test])
        
        # The remaining 30% are for the test set
        test_set.append(group.iloc[-n_test:])

    # Concatenate all the example and test data back into DataFrames
    example_set = pd.concat(example_set).reset_index(drop=True)
    test_set = pd.concat(test_set).reset_index(drop=True)

    return example_set, test_set

def write_prompt(prompt_type, speech_amr=None, gesture_amrs=None, gesture_labels=None, examples=None):
    """
    Write prompt based on the specific condition.
    Args:
        prompt_type: speech, gesture, or speech_gesture
        speech_amr: speech amr retrieved from test set, Llama generates a sentence for this amr
        gesture_amrs: gesture amr(s) retrieved from test set, Llama generates a sentence for this amr
        gesture_labels: gesture label(s) retrieved from test set, Llama generates a sentence for these labels
        examples: list of examples shown to Llama to learn the AMR format
    Returns:
        prompt: the generated prompt
    """
    if prompt_type == "speech":
        prompt = "Given the following speech Abstract Meaning Representation (AMR), generate a corresponding sentence in natural spoken English. Provide a short explanation."
        if examples:
            prompt += " First, read the examples to understand the Abstract Meaning Representation format:\n"
            for i, ex in enumerate(examples):
                prompt += f"\nExample {i+1}:\n"
                prompt += f"Sentence: {ex['sentence']}\n"
                prompt += f"Speech AMR:\n {pretty_amr(ex['speech_amr'])}\n"
                
            prompt += "\nNow, generate a sentence from the following speech AMR and explain your reasoning. Please provide the output only in json format:\n"
            prompt += '[{"sentence": "Your generated sentence here.", "explanation": "Your explanation here."}]\n'
            prompt += f"Speech AMR:\n {pretty_amr(speech_amr)}\n"
            return prompt
            
    elif prompt_type == "gesture":
        prompt = "Given the following gesture label(s) and gesture Abstract Meaning Representation (AMR), generate a corresponding sentence and speech AMR in natural spoken English. Provide a short explanation."
        if examples:
            prompt += " First, read the examples to understand the Abstract Meaning Representation format:\n"
            for i, ex in enumerate(examples):
                prompt += f"\nExample {i+1}:\n"
                prompt += f"Sentence: {ex['sentence']}\n"
                prompt += f"Speech AMR:\n {pretty_amr(ex['speech_amr'])}\n"
                prompt += f"\nGesture label(s): \n" + "\n".join(ex['gesture_labels']) + "\n"
                prompt += f"\nGesture AMR:\n"  
                for amr in ex["gesture_amrs"]:
                    prompt += pretty_amr(amr) + "\n"
                    
            prompt += "\nNow, generate a sentence and the corresponding speech AMR from the following gesture label(s) and gesture AMR and explain your reasoning. Please provide the output only in json format. The speech AMR should be on one line:\n"
            prompt += '[{"sentence": "Your generated sentence here.", "speech AMR:" "Your generated speech AMR here", "explanation": "Your explanation here."}]\n'
            prompt += f"Gesture label(s):\n" + "\n".join(gesture_labels) + "\n"
            prompt += f"Gesture AMR:\n"      
            for amr in gesture_amrs:
                    prompt += pretty_amr(amr) + "\n"
            return prompt

    elif prompt_type == "speech_gesture":
        prompt = "Given the following gesture label(s), gesture Abstract Meaning Representation (AMR) and speech AMR, generate a corresponding sentence in natural spoken English. Provide a short explanation."
        if examples:
            prompt += " First, read the examples to understand the Abstract Meaning Representation format:\n"
            for i, ex in enumerate(examples):
                prompt += f"\nExample {i+1}:\n"
                prompt += f"Sentence: {ex['sentence']}\n"
                prompt += f"Speech AMR:\n {pretty_amr(ex['speech_amr'])}\n"
                prompt += f"\nGesture label(s): \n" + "\n".join(ex['gesture_labels']) + "\n"
                prompt += f"\nGesture AMR:\n" 
                for amr in ex["gesture_amrs"]:
                    prompt += pretty_amr(amr) + "\n" 
                    
            prompt += "\nNow, generate a sentence from the following gesture label(s), gesture AMR and speech AMR and explain your reasoning. Please provide the output only in json format:\n"
            prompt += '[{"sentence": "Your generated sentence here.", "explanation": "Your explanation here."}]\n'
            prompt += f"Gesture label(s):\n" + "\n".join(gesture_labels) + "\n"
            prompt += f"Gesture AMR:\n"     
            for amr in gesture_amrs:
                    prompt += pretty_amr(amr) + "\n"
            prompt += f"Speech AMR:\n {pretty_amr(speech_amr)}\n"
            return prompt
    else:
        raise ValueError(f"Invalid prompt_type: {prompt_type}")

def generate_prompt(df_test, examples):
    """
    Generate the prompt per condition.
    Args:
       df_test: dataframe of the test set
       examples: list of examples
    Returns:
        prompts: list of dictionaries containing the prompt, the reference
        sentence, condition type (referred to as scenario), and file name
    """
    prompts = []
    
    example_idx = 0

    for idx, row in df_test.iterrows():
        sampled_examples = [examples[example_idx], examples[(example_idx + 1) % len(examples)]]    ## Select two examples to give to Llama
        example_idx = (example_idx + 2) % len(examples)                                            ## Increase idx, once out of range repeat from the beginning               
                                                 
        prompt = write_prompt(
            prompt_type=row["prompt_type"],                       
            speech_amr=row.get("speech_amr"),
            gesture_amrs=row.get("gesture_amrs"),
            gesture_labels=row.get("gesture_labels"),
            examples=sampled_examples  
        )
    
        prompts.append({
            "prompt": prompt,
            "meta": {
                "sentence": row["sentence"],
                "scenario": row["prompt_type"],
                "file": row["file"]
            }
        })
    return prompts

def write_prompt_error_anaysis(prompt_type, speech_amr=None, gesture_amrs=None, gesture_labels=None, examples=None):
    """
    Write prompt based on the specific condition, different prompts for the error analysis
    Args:
        prompt_type: speech, gesture, or speech_gesture
        speech_amr: speech amr retrieved from test set, Llama generates a sentence for this amr
        gesture_amrs: gesture amr(s) retrieved from test set, Llama generates a sentence for this amr
        gesture_labels: gesture label(s) retrieved from test set, Llama generates a sentence for these labels
        examples: list of examples shown to Llama to learn the AMR format
    Returns:
        prompt: the generated prompt
    """
    
    if prompt_type == "speech":
        prompt = "You are interpreting verbal communication in a collaborative block-building task. One participant is explaining how to construct a specific block structure using both speech and gestures. In this task, you are given only the speech Abstract Meaning Representation:\nSpeech AMR: A structured semantic representation of what was spoken, consisting of Propbank framesets and arguments.\nYour task is to:\n1. Generate a sentence in natural spoken English using speech AMR that reflects what the speaker is trying to communicate.\n2. Provide a short explanation describing how you interpreted the speech AMR to generate the sentence.\n"
        if examples:
            prompt += "First, read the examples to understand the Abstract Meaning Representation format:\n"
            for i, ex in enumerate(examples):
                prompt += f"\nExample {i+1}:\n"
                prompt += f"Sentence: {ex['sentence']}\n"
                prompt += f"Speech AMR:\n {pretty_amr(ex['speech_amr'])}\n"
                
            prompt += "\nNow, generate a sentence from the following speech AMR and explain your reasoning. Please provide the output only in json format:\n"
            prompt += '[{"sentence": "Your generated sentence here.", "explanation": "Your explanation here."}]\n'
            prompt += f"Speech AMR:\n {pretty_amr(speech_amr)}\n"
            return prompt
            
    elif prompt_type == "gesture":
        prompt = "You are interpreting gesture information from a collaborative block-building task. One participant is explaining how to construct a specific block structure using both speech and gestures. In this task, you are given only the gesture information:\nGesture Labels: Descriptions of hand, arm, or body movements.\nGesture AMR: A structured semantic representation of gestures used.\nAlthough the participant was also speaking, only the gesture information is provided here. Your task is to:\n1. Generate a sentence in natural spoken English using gesture information that reflects what the speaker was likely trying to communicate with the gestures.\n2. Provide a short explanation describing how you interpreted the gesture information to generate the sentence.\n"
        if examples:
            prompt += "First, read the examples to understand the Abstract Meaning Representation format and the gesture labels:\n"
            for i, ex in enumerate(examples):
                prompt += f"\nExample {i+1}:\n"
                prompt += f"Sentence: {ex['sentence']}\n"
                prompt += f"Speech AMR:\n {pretty_amr(ex['speech_amr'])}\n"
                prompt += f"\nGesture label(s): \n" + "\n".join(ex['gesture_labels']) + "\n"
                prompt += f"\nGesture AMR:\n"   
                for amr in ex["gesture_amrs"]:
                    prompt += pretty_amr(amr) + "\n"
                    
            prompt += "\nNow, generate a sentence and the corresponding speech AMR from the following gesture label(s) and gesture AMR and explain your reasoning. Please provide the output only in json format. The speech AMR should be on one line:\n"
            prompt += '[{"sentence": "Your generated sentence here.", "speech AMR:" "Your generated speech AMR here", "explanation": "Your explanation here."}]\n'
            prompt += f"Gesture label(s):\n" + "\n".join(gesture_labels) + "\n"
            prompt += f"Gesture AMR:\n"      
            for amr in gesture_amrs:
                    prompt += pretty_amr(amr) + "\n"
            return prompt

    elif prompt_type == "speech_gesture":
        prompt = "You are interpreting multimodal communication in a collaborative block-building task. One participant is verbally and gesturally explaining how to construct a specific block structure. Your input includes:\nGesture Labels: Descriptions of hand, arm, or body movements.\nGesture AMR: A structured semantic representation of gestures used.\nSpeech AMR: A structured semantic representation of what was spoken, consisting of Propbank framesets and arguments.\nYour task is to:\n1. Generate a sentence in natural spoken English using speech and gesture information that reflects what the speaker is trying to communicate.\n2. Provide a short explanation describing how you interpreted both the gesture and speech AMRs and the gesture labels to generate the sentence.\n"
        if examples:
            prompt += "First, read the examples to understand the Abstract Meaning Representation format and the gesture labels:\n"
            for i, ex in enumerate(examples):
                prompt += f"\nExample {i+1}:\n"
                prompt += f"Sentence: {ex['sentence']}\n"
                prompt += f"Speech AMR:\n {pretty_amr(ex['speech_amr'])}\n"
                prompt += f"\nGesture label(s): \n" + "\n".join(ex['gesture_labels']) + "\n"
                prompt += f"\nGesture AMR:\n"  
                for amr in ex["gesture_amrs"]:
                    prompt += pretty_amr(amr) + "\n" 
                    
            prompt += "\nNow, generate a sentence from the following gesture label(s), gesture AMR and speech AMR and explain your reasoning. Please provide the output only in json format:\n"
            prompt += '[{"sentence": "Your generated sentence here.", "explanation": "Your explanation here."}]\n'
            prompt += f"Gesture label(s):\n" + "\n".join(gesture_labels) + "\n"
            prompt += f"Gesture AMR:\n"     
            for amr in gesture_amrs:
                    prompt += pretty_amr(amr) + "\n"
            prompt += f"Speech AMR:\n {pretty_amr(speech_amr)}\n"
            return prompt
    
    else:
        raise ValueError(f"Invalid prompt_type: {prompt_type}")

def generate_prompt_error_anaysis(df_test, examples):
    """
    Generate the prompt per condition for the error analysis.
    Args:
       df_test: dataframe of the test set
       examples: list of examples
    Returns:
        prompts: list of dictionaries containing the prompt, the reference
        sentence, condition type (referred to as scenario), and file name
    """
    prompts = []
    
    example_idx = 0

    for idx, row in df_test.iterrows():
        sampled_examples = [examples[example_idx], examples[(example_idx + 1) % len(examples)]] 
        example_idx = (example_idx + 2) % len(examples)
        
        prompt = write_prompt_error_anaysis(
            prompt_type=row["prompt_type"],
            speech_amr=row.get("speech_amr"),
            gesture_amrs=row.get("gesture_amrs"),
            gesture_labels=row.get("gesture_labels"),
            examples=sampled_examples  
        )
    
        prompts.append({
            "prompt": prompt,
            "meta": {
                "sentence": row["sentence"],
                "scenario": row["prompt_type"],
                "file": row["file"]
            }
        })
    return prompts

def query_LLM(model_client, prompt, temp=0.3):
    """
    Function that prompts Llama.
    Args:
        model_client: points to the server. In this thesis it's "OpenAI(base_url="http://localhost:8000/v1", api_key="cltl")"
        prompt: prompt to give to Llama
        temp: temperature, set to 0.3
    Returns:
        completion: Llama's generated output
    """
    history = [
        {"role": "system", "content": prompt},
    ]
    completion = model_client.chat.completions.create(
        model="local-model", 
        messages=history,
        temperature=temp,
        #max_tokens=2048,
        top_p=0.9,
        stream=False,
        seed=2201
    )
    return completion.choices[0].message.content

def query_LLM_multiple(model_client, prompt, temp=0.3, n=3):
    """
    Function that prompts Llama three times with different seeds.
    Args:
        model_client: points to the server. In this thesis it's "OpenAI(base_url="http://localhost:8000/v1", api_key="cltl")"
        prompt: prompt to give to Llama
        temp: temperature, set to 0.3
        n: number of times to prompt Llama, set to three
    Returns:
        completions: Llama's generated outputs
    """
    completions = []
    for i in range(n):
        response = model_client.chat.completions.create(
            model="local-model",  
            messages=[{"role": "system", "content": prompt}],
            temperature=temp,
            max_tokens=2048,
            stream=False,
            seed=13 + i                    ## Slightly vary seed for different outputs
        )
        completions.append(response.choices[0].message.content)
    return completions

def count_error_types(outputfile):
    """
    Function that calculates how many instances each error type has.
    Args:
        outputfile: file containing the annotated sentences
    Returns:
        deictic, propbank, polarity, gesture, 
        amr, incoherent, other, no_mistake, total: 
        value per error type showcasing the number of errors per error type
    """
    propbank = 0
    polarity = 0
    deictic = 0
    gesture = 0
    amr = 0
    incoherent = 0
    other = 0
    no_mistake = 0
    total = 0

    with open(outputfile, "r", encoding="utf-8") as file:
        for line in file:
            line = line.strip().lower()
            if not line:
                continue
            if 'deictic' in line:
                deictic += 1
                total += 1
            if 'propbank' in line:
                propbank += 1
                total += 1
            if 'polarity' in line:
                polarity += 1
                total += 1
            if 'gesture' in line:
                gesture += 1
                total += 1
            if 'amr' in line:
                amr += 1
                total += 1
            if 'incoherent' in line:
                incoherent += 1
                total += 1
            if 'other' in line:
                other += 1
                total += 1
            if '0' in line:
                no_mistake += 1
    
    return deictic, propbank, polarity, gesture, amr, incoherent, other, no_mistake, total