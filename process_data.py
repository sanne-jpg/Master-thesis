import pandas as pd
import re

def extract_data_from_txt(file_path):
    """
    Extracts data from the provided text file: begin and end time of the
    utterance, as well as the corresponding speech and gesture AMR and 
    gesture labels.
    
    Args:
        file_path (str): Path to the data files.
    Returns:
        A list of dicionaries, containing the begin time, end time, sentence,
        corresponding speech AMR, as well as gesture label(s) and gesture AMR.
    """
    data = []
    with open(file_path, "r", encoding = "utf-8") as file:
        for line in file.readlines():
            parts = line.strip().split("\t")
            if parts[0].startswith("Begin"):
                continue
            
            ## Extract speech AMR
            if len(parts) > 3 and parts[3].startswith("("):  
                joined_speech_amr = " ".join(parts[3:])
                parts = parts[:3] + [joined_speech_amr]             ## Replace everything from index 3 onward with the single joined AMR

            ## Extract gesture AMR
            if len(parts) > 5 and parts[5].startswith("("):     
                joined_gesture_amr = " ".join(parts[5:])
                parts = parts[:5] + [joined_gesture_amr]
            
            if len(parts) >= 3:  # Ensure valid data                           
                begin_time, end_time = float(parts[0]), float(parts[1])
                speech = parts[2] if len(parts) == 3 else ""
                speech_amr = parts[3] if len(parts) >= 4 else ""
                gesture_label = parts[4] if len(parts) == 5 else ""
                gesture_amr = parts[5] if len(parts) == 6 else ""
                
                data.append({
                    "begin_time": begin_time,
                    "end_time": end_time,
                    "speech": speech,
                    "speech_amr": speech_amr,
                    "gesture_label": gesture_label,
                    "gesture_amr": gesture_amr
                })
    return data

def merge_speech_and_gesture_with_amr(data, filename=None, margin=0.017):
    """
    Function that aligns the spoken sentence with the speech AMR,
    gesture AMR, and gesture labels.
    Args:
        data: output list from def extract_data_from_txt 
        filename: the specific file that is considered
        margin: margin to account for sentences and AMR that do not
        start or end exactly at the same time
    Returns:
        grouped: list of dictionaries containing file name, begin
        and end time, the spoken sentence, along with its corresponding
        speech AMR, gesture AMR, and gesture label(s)
    """
    speech_amrs = [entry for entry in data if entry["speech_amr"]]
    speech_words = [entry for entry in data if entry["speech"]]
    gesture_inst = [entry for entry in data if entry["gesture_amr"] or entry["gesture_label"]]

    grouped = []

    for amr_entry in speech_amrs:
        start_amr = amr_entry["begin_time"] - margin        
        end_amr = amr_entry["end_time"] + margin

        matching_words = [
            w for w in speech_words                                                  ## When multiple words falls in between the same time stamps of the speech_amr,
            if w["begin_time"] >= start_amr and w["begin_time"] <= end_amr           ## they should be grouped together        
        ]  
        
        full_sentence = " ".join([w["speech"] for w in matching_words]).strip()      ## Join the words into a sentence
        full_sentence = re.sub(r'\s+', ' ', full_sentence)                           ## Remove any extra spaces
        
        matching_gestures = [
            g for g in gesture_inst                                                                 ## When label(s) falls in between the same time stamps of the speech_amr,
            if g["gesture_label"] and g["begin_time"] <= end_amr and g["end_time"] >= start_amr     ## they're grouped together
        ]
        gesture_labels = [g["gesture_label"] for g in matching_gestures]             ## Extract the labels
        
        matching_gesture_amrs = [
            g for g in gesture_inst                                                                 ## Do the same for the gesture AMR
            if g["gesture_amr"] and g["begin_time"] <= end_amr and g["end_time"] >= start_amr
        ]
        gesture_amrs = [g["gesture_amr"] for g in matching_gesture_amrs]             ## Extract the gesture AMR
        
        grouped.append({      
            "file": filename.replace('.txt', ''),
            "begin_time": round(amr_entry["begin_time"], 2),                      
            "end_time": round(amr_entry["end_time"], 2),                          
            "sentence": full_sentence,
            "speech_amr": amr_entry["speech_amr"],
            "gesture_labels": gesture_labels,
            "gesture_amrs": gesture_amrs
        })

    return grouped

def create_dataset(input, outputfile, source_name=None, mode="w"):
    """
    Function that creates the dataset by writing the input to file.
    
    Args:
        input: list containing the needed items
        outputfile: path to the outputfile
        mode: writing mode. First "w", to add the rest it should be "a"
    """
    with open(outputfile, mode, encoding='utf8') as outfile:
        i = 1
        if source_name:
            outfile.write(f"File: {source_name}\n")
        for dict in input:
            parts = []
            parts.append(f'speech {i}: "{dict["sentence"]}"\n')

            # Speech AMR
            parts.append(f'speech_amr {i}:')         
            parts.append(pretty_amr(dict["speech_amr"]))
    
            # Gesture labels
            parts.append(f'\ngesture_labels {i}:')      
            if dict["gesture_labels"]:
                for label in dict["gesture_labels"]:
                    parts.append(label)
            else:
                parts.append("None")
    
            # Gesture AMRs
            parts.append(f'\ngesture_amr {i}:')             
            if dict["gesture_amrs"]:
                for amr in dict["gesture_amrs"]:
                    parts.append(pretty_amr(amr))  # Function to format AMRs
            else:
                parts.append("None")
    
            parts.append("\n\n")
        
            print("\n".join(parts))
            outfile.write("\n".join(parts))
            i += 1

def pretty_amr(raw_amr):
    """
    Format the input as pretty, readable AMR

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

def count_statistics(all_merged_data):
    """
    Function that calculates some statistics over the dataset.
    Args:
        all_merged_data: list containing all dataset information
        (file name, begin and end time, sentence, corresponsing
        speech AMR, gesture AMR, and gesture label(s))
    """
    ## Number of sentences
    sent_count = 0
    for item in all_merged_data:
        if item['file']:
            sent_count += 1

    ## Number of sentences that have gestures
    sent_with_gestures = 0
    for item in all_merged_data:
        if item['gesture_amrs'] != []:
            sent_with_gestures += 1

    ## Average sentence length
    total_tokens = 0
    for item in all_merged_data:
        if item['gesture_amrs'] != []:
            #print(item['gesture_amrs']) 
            sent = item['sentence']
            sent_tokenized = sent.split()
            total_tokens += len(sent_tokenized)
            avg_length = round(total_tokens / sent_without_gestures, 2)

    ## How many gestures of each type
    icon_count = 0
    deixis_count = 0
    emblem_count = 0
    for item in all_merged_data:
        for gesture in item['gesture_amrs']:
            icon_count += gesture.count("icon")
            deixis_count += gesture.count("deixis")
            emblem_count += gesture.count("emblem")
    
    return sent_count, sent_with_gestures, avg_length, icon_count, deixis_count, emblem_count

def clean_data_for_prompting(all_merged_data):
    """
    Remove time stamps and add num_gesture_amrs to each item for clarity.
    Args:
        all_merged_data: list containing all dataset information
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