import config
import shutil
import os
import sys
import json
import re
import random
from pathlib import Path

import file_writer
from qa_modules import QA_NoContext_deepseek_V3,QA_NoContext_deepseek_R1

# import prompt
from pydantic import BaseModel, Field
from typing import List, Dict
from langchain.output_parsers import PydanticOutputParser

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def robust_llm_parse(qa, prompt, parser, max_retries=3):
    """General LLM response parsing function with automatic retries and fallback result.
    Args:
        qa (object): LLM interface object
        prompt (str): Prompt string sent to the LLM
        parser (object): Parser object for parsing the LLM response
        fallback (object, optional): Alternative result to return if parsing fails (e.g., random selection, empty dict)
        max_retries (int, optional): Maximum number of retry attempts
    Returns:
        result (object): Parsed result if successful; fallback if all retries fail.
    """
    for attempt in range(max_retries):
        try:
            response = qa.ask(prompt)
            result = parser.parse(response)
            return result
        except Exception as e:
            print(f"LLM response parsing failed (attempt {attempt+1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                continue
            else:
                print("Maximum retry attempts reached, using fallback result")
                return None


def select_random_items(a, number):
    # Filter out key-value pairs where value string length exceeds 10000
    filtered_a = {k: v for k, v in a.items() if len(str(v)) <= 10000}
    
    # Check if processed dictionary has more than 5 keys
    if len(filtered_a) > number:
        # Randomly select 5 keys
        selected_keys = random.sample(list(filtered_a.keys()), number)
        # Build new dictionary
        return {key: filtered_a[key] for key in selected_keys}
    else:
        return filtered_a
        
    
def dict_to_json_string(data):
    json_str = json.dumps(data, ensure_ascii=False, indent=2)
    return json_str

def create_OF_case_json(source_dir):
    """Save OpenFOAM case directory (0/, constant/, system/) file contents to JSON document
    Args:
        source_dir: Case root directory path (folder containing 0/ constant/ system/)
        output_json: Output JSON file path
    Returns:
        None: results will be written to specified JSON file
    """
    file_data = {}
    
    # Three target directories to process
    target_dirs = ["0", "constant", "system"]
    
    for dir_name in target_dirs:
        dir_path = os.path.join(source_dir, dir_name)
        
        # Skip if directory doesn't exist
        if not os.path.isdir(dir_path):
            continue
            
        # Traverse entries in directory
        for entry in os.listdir(dir_path):
            entry_path = os.path.join(dir_path, entry)
            
            # Only process files (ignore subdirectories)
            if os.path.isfile(entry_path):
                # Build path relative to source directory (using POSIX style paths)
                relative_path = os.path.join(dir_name, entry).replace("\\", "/")
                
                try:
                    # Read file content
                    with open(entry_path, "r", encoding="utf-8") as f:
                        content = f.read()
                        file_data[relative_path] = content
                except Exception as e:
                    print(f"Unable to read file {entry_path}: {str(e)}")
                    continue

    return dict_to_json_string(file_data)


def list_case_file(case_path):
    target_folders = ['0', 'system', 'constant']
    file_list = []
    for folder in target_folders:
        folder_path = os.path.join(case_path, folder)
        if not os.path.isdir(folder_path):
            continue
        for entry in os.listdir(folder_path):
            entry_path = os.path.join(folder_path, entry)
            if os.path.isfile(entry_path):
                # Build relative path and unify to Linux-style path separators
                rel_path = f"{folder}/{entry}"
                file_list.append(rel_path)
    return file_list

def identify_error_to_add_new_file(running_error, relevant_reflections=""):

    analyze_error_to_add_new_file = f'''{config.general_prompts}. OpenFOAM File Requirement Analyzer
Analyze the runtime error {running_error} to:

1. Check if it contains the exact phrase "cannot find file"
2. If present:
a) Identify the missing file path from the error message
b) Format the response as 0/..., system/..., or constant/...
3. If absent/irrelevant: Respond with no

1. Respond ONLY with a filename (format: 0/xx, system/xx, or constant/xx) if required
2. Respond ONLY with 'no' if no file needed
3. Strict formatting requirements:
- No special characters: (), '', ", `
- No markdown/formatting symbols
- No whitespace, line breaks, or indentation
- No explanations or extra text
Your response must be exactly one of:
a) A directory-path formatted string from allowed locations
b) The lowercase string 'no'
Examples of valid responses:
system/fvSchemes
constant/g
no'''
    
    if relevant_reflections != "":
        analyze_error_to_add_new_file += f"\n{relevant_reflections}"

    qa = QA_NoContext_deepseek_V3()

    answer = qa.ask(analyze_error_to_add_new_file)

    pure_response = file_writer.extract_pure_response(answer)

    return pure_response

def read_files_to_dict(base_dir):
    file_dict = {}
    # List of directories to traverse
    target_dirs = ['0', 'system', 'constant']
    
    for dir_name in target_dirs:
        dir_path = os.path.join(base_dir, dir_name)
        # Skip if directory doesn't exist
        if not os.path.isdir(dir_path):
            continue
        # Traverse all entries under directory
        for entry in os.listdir(dir_path):
            full_path = os.path.join(dir_path, entry)
            # Only process files, ignore subdirectories
            if os.path.isfile(full_path):
                # Build relative path as key (relative to base_dir)
                rel_path = os.path.join(dir_name, entry)
                # Read file content
                try:
                    with open(full_path, 'r', encoding='utf-8') as f:
                        file_dict[rel_path] = f.read()
                except Exception as e:
                    file_dict[rel_path] = f"<Error reading file: {str(e)}>"
    
    return file_dict

def add_new_file(file_name):

    print(f"adding new file: {file_name}")

    file_path = f'{config.OUTPUT_PATH}/{file_name}'

    other_case_file_content = read_files_to_dict(config.OUTPUT_PATH)

    add_new_file_prompt = f'''
    A new case file {file_name} must be add to the OpenFOAM case dir. The file contents of other case files are: {other_case_file_content}. Please respond the file contents for the new file which can make this case run correctly with other case files. Ensure the dimension is correct if the dimension shows in the file content.

    In your response: Absolutely AVOID any elements including but not limited to:
    - Markdown code block markers (``` or  ```)
    - Extra comments or explanations
    - Unnecessary empty lines or indentation
    '''

    qa = QA_NoContext_deepseek_R1()

    answer = qa.ask(add_new_file_prompt)

    # Perform dimension correction
    file_content = answer
    with open(os.path.join(config.path_cfg.database_dir, 'OF_case_dimensions.json'), 'r', encoding='utf-8') as f:
        dimensions_dict = json.load(f)
    if file_name in dimensions_dict.keys():
        if "dimensions" in file_content:
            dimension_pattern = r'dimensions.*?\n'
            match = re.search(dimension_pattern, file_content)
            if match:
                if file_name+"_" in dimensions_dict.keys(): # The underlined parts cannot be compressed.
                    if file_name in ["0/p", "0/p_rgh", "0/alphat"]: 
                        if config.case_info.case_solver in config.compressible_solvers or "compressible" in file_content:
                            file_content = re.sub(dimension_pattern, f'dimensions      {dimensions_dict[file_name]};\n', file_content)
                        elif config.case_info.case_solver in config.incompressible_solvers:
                            file_content = re.sub(dimension_pattern, f'dimensions      {dimensions_dict[file_name+"_"]};\n', file_content)
                        else:
                            pass
                    else:
                        pass # Skip decision, let LLM set according to initially generated content
                else:
                    file_content = re.sub(dimension_pattern, f'dimensions      {dimensions_dict[file_name]};\n', file_content)
        else:
            if file_name+"_" in dimensions_dict.keys(): 
                if file_name in ["0/p", "0/p_rgh", "0/alphat"]: 
                    if config.case_info.case_solver in config.compressible_solvers or "compressible" in file_content:
                        file_content = f"\ndimensions      {dimensions_dict[file_name]};\n"
                    elif config.case_info.case_solver in config.incompressible_solvers:
                        file_content = f"\ndimensions      {dimensions_dict[file_name+'_']};\n"
                else:
                    pass    # File will error, let LLM solve it itself
            else:
                file_content += f"\ndimensions      {dimensions_dict[file_name]};\n"

    try:
        file_writer.write_field_to_file(file_content,file_path)
        print(f"write the file {file_name}")
    except Exception as e:
        print(f"Errors occur during write_field_to_file: {e}")
    else: # Successfully executed field file write operation
        file_write_successful = True
    return answer

def identify_file_name_from_error(running_error):

    case_files = list_case_file(config.OUTPUT_PATH)

    analyze_running_error_prompt = f'''
    Analyze the provided OpenFOAM runtime error { {running_error} } to identify the file requires revision. The result must be one of the following files: { {case_files} }. You response must only include the case name.

    In your response: Absolutely AVOID any elements including but not limited to:
    - Markdown code block markers (``` or ```)
    - Extra comments or explanations
    - Unnecessary empty lines or indentation
    '''
    
    qa = QA_NoContext_deepseek_R1()

    answer = qa.ask(analyze_running_error_prompt)

    answer = answer.strip()

    # file_for_revision = file_writer.extract_pure_response(answer)

    return answer

# Parameter target_file is the file name identified from running_error that needs to be modified
# Find by solver
def find_reference_files_by_solver(target_file):
    case_solver = config.case_solver
    turbulence_model = config.case_turbulence_model
    turbulence_model_list = [
        "SpalartAllmarasIDDES",
        "SpalartAllmarasDDES",
        "SpalartAllmaras",
        "kEpsilon",
        "WALE",
        "kEqn",
        "LaunderSharmaKE",
        "realizableKE",
        "kOmegaSSTLM",
        "RNGkEpsilon",
        "buoyantKEpsilon",
        "kkLOmega",
        "PDRkEpsilon",
        "kOmegaSST",
        "dynamicKEqn"
    ]
    if turbulence_model not in turbulence_model_list:
        turbulence_model = None

    other_physical_model = config.other_physical_model

    # 返回内容
    target_file_reference = {}

    solver_type = None

    file_number = 0

    for key, value in config.OF_case_data_dict.items():
        if case_solver in key and turbulence_model == value["turbulence_model"]:
            
            # If there is other_physical_model, need to check if they are the same
            if "other_physical_model" in value.keys():
                if other_physical_model == None or other_physical_model == []:
                    if value["other_physical_model"] != [] and value["other_physical_model"] != None and value["other_physical_model"] != ["common"]:
                        continue  
                else:
                    if not set(other_physical_model) == set(value["other_physical_model"]):
                        continue

            config_files = value['configuration_files']
            if target_file in config_files.keys():
                # print(config_files.keys())
                new_file_key = f'sample_file_{file_number}'
                file_number += 1
                target_file_reference[new_file_key] = config_files[target_file]
    
    # If not found, do not consider turbulence model matching
    if file_number == 0:
        for key, value in config.OF_case_data_dict.items():
            if case_solver in key:
                config_files = value['configuration_files']

                if target_file in config_files.keys():
                    # print(config_files.keys())
                    new_file_key = f'sample_file_{file_number}'
                    file_number += 1
                    target_file_reference[new_file_key] = config_files[target_file]

    # If the above result is 0, search at a higher level, first find the solver type, such as compressible
    if file_number == 0:
        for key, value in config.OF_case_data_dict.items():
            if case_solver in key:
                path_split = key.split('/')
                solver_type = path_split[0]
                break

    # Find target_file under the solver type
    if solver_type is not None:
        for key, value in config.OF_case_data_dict.items():
                path_split = key.split('/')
                if solver_type == path_split[0]:
                    config_files = value['configuration_files']
                    if target_file in config_files.keys():
                        # print(config_files.keys())
                        new_file_key = f'sample_file_{file_number}'
                        file_number += 1
                        target_file_reference[key] = config_files[target_file]

    # print(target_file_reference.keys())
    target_file_reference = {k: v for k, v in target_file_reference.items() if v != ""} # Remove empty values

    several_target_file_reference = select_random_items(target_file_reference, 3)

    return dict_to_json_string(several_target_file_reference)

def analyze_running_error_with_all_case_file_content(running_error):
    """Find the file causing the error and its modification suggestions"""

    all_case_file_content = create_OF_case_json(config.OUTPUT_PATH)

    file_content = None

    case_files = list_case_file(config.OUTPUT_PATH)

    case_files = dict_to_json_string(case_files)    # json.dumps()

    analyze_running_error_prompt = f'''
    Analyze the provided OpenFOAM runtime error { {running_error} } to identify the root cause and which case file needs to be revised to fix the runtime error. The OpenFOAM case files are given as json-format string as { {all_case_file_content} }. 
    Your response must be a json format string with following keys and values:
    - a 'wrong_file' key, and its value the file name which will be revised to fix the error. This file name must be one of the case files { {case_files} }.
    - a 'advices_for_revision' key, and its value provide a step-by-step fix of the 'wrong_file' to fix the error. Ensure the advice addresses the error’s technical cause (e.g., CFL violation, invalid discretization scheme, missing required keyword). The advice must be a string.

    In your JSON response: Absolutely AVOID any elements including but not limited to:
    - Markdown code block markers (``` or ```)
    - Extra comments or explanations
    - Unnecessary empty lines or indentation
    '''
    
    qa = QA_NoContext_deepseek_R1()

    answer = qa.ask(analyze_running_error_prompt)

    answer = answer.strip("```")
    if answer.startswith("json\n"):
        answer = answer[len("json\n"):]
    dict_answer = json.loads(answer.strip("```"))

    advices_for_revision = dict_answer['advices_for_revision']
    wrong_file = dict_answer['wrong_file']

    return [wrong_file,advices_for_revision]

def analyze_running_error_2(running_error, file_name):

    file_content = None

    file_path = f'{config.OUTPUT_PATH}/{file_name}'

    with open(file_path, "r", encoding="utf-8") as file:
        file_content = file.read()

    case_files = list_case_file(config.OUTPUT_PATH)

    analyze_running_error_prompt = f'''
    Analyze the provided OpenFOAM runtime error {running_error} to identify the root cause. Give advice on correcting the file {file_name} with the file contents as {file_content}.

    In your response: Provide a step-by-step fix (e.g., adjust endTime, modify tolerance in solver settings, correct boundary type in U). Ensure the advice addresses the error’s technical cause (e.g., CFL violation, invalid discretization scheme, missing required keyword). The advice must be a string.

    In your response: Absolutely AVOID any elements including but not limited to:
    - Markdown code block markers (``` or ```)
    - Extra comments or explanations
    - Unnecessary empty lines or indentation
    '''
    
    qa = QA_NoContext_deepseek_R1()

    answer = qa.ask(analyze_running_error_prompt)

    advices_for_revision = answer

    return advices_for_revision

def analyze_error_repetition(error_history):
    answer = 'no'
    if(len(error_history)>=3):
        error_minus_1 = error_history[-1]
        error_minus_2 = error_history[-2]
        error_minus_3 = error_history[-3]

        analyze_running_error_repetition_prompt = f'''
        Analyze the following three error histories to identify whether the error have repetitively shown three time. Error 1: {error_minus_1}. Error 2: {error_minus_2}. Error 3: {error_minus_3}. If the error have repetitively shown three times, respond 'yes'; otherwise, respond 'no'. You must only respond 'yes' or 'no'.
        '''

        qa = QA_NoContext_deepseek_R1()

        answer = qa.ask(analyze_running_error_repetition_prompt)

    answer = answer.strip()

    if answer.lower() == 'yes':
        return True
    else:
        return False

def rewrite_file(file_name, reference_files):
    print(f"rewriting {file_name}")

    file_content = None

    file_path = f'{config.OUTPUT_PATH}/{file_name}'

    with open(file_path, "r", encoding="utf-8") as file:
        file_content = file.read()

    correct_file_prompt = f'''{config.general_prompts}
    Please rewrite the {file_name} file for the OpenFOAM case. The original file content is: {file_content}. You can reference these files from OpenFOAM tutorial {reference_files} for formating and key values. Ensure the dimension is correct if the dimension shows in the file content.

    In your response: Absolutely AVOID any elements including but not limited to:
    - Markdown code block markers (``` or  ```)
    - Extra comments or explanations
    '''

    qa = QA_NoContext_deepseek_V3()

    answer = qa.ask(correct_file_prompt)

    answer = file_writer.extract_pure_response(answer)

    try:
        file_writer.write_field_to_file(answer,file_path)
        print(f"write the file {file_name}")

    except Exception as e:
        print(f"Errors occur during write_field_to_file: {e}")
    else: # Successfully executed field file write operation
        file_write_successful = True


def analyze_running_error_with_reference_files(running_error, file_name,early_revision_advice, reference_files):

    file_content = None

    file_path = f'{config.OUTPUT_PATH}/{file_name}'

    with open(file_path, "r", encoding="utf-8") as file:
        file_content = file.read()

    case_files = list_case_file(config.OUTPUT_PATH)

    analyze_running_error_prompt = f'''Analyze the provided OpenFOAM runtime error [[[ {running_error} ]]] to identify the root cause. Give advice on correcting the file { {file_name} } with the file contents as [[[ {file_content} ]]]. The revision must not alter the file to voilate these initial and boundary conditions in the paper [[[ {config.case_ic_bc_from_paper} ]]]. You can refer to these files from OpenFOAM tutorial [[[ {reference_files} ]]] to improve the correction advice.

In your response: Provide a step-by-step fix. Ensure the advice addresses the error's technical cause. The advice must be a string.
Additionally, if an error occurs due to keyword settings, you need to analyze whether the content corresponding to that keyword should be set in { {file_name} }. If it is determined that such content should not appear in { {file_name} }, you should explicitly point it out and recommend its removal(common in files within the constant folder).For example, if the {{specie}} keyword in the {{thermo.compressibleGas}} file is incorrect, analysis shows that {{specie}} belongs to the {{mixture}} section of the file. However, {{mixture}} should be set in {{thermophysicalProperties}}, not in {{thermo.compressibleGas}}. So, it is recommended to delete the settings related to mixture.

In your response: Absolutely AVOID any elements including but not limited to:
- Markdown code block markers (``` or ```)
- Extra comments or explanations
- Unnecessary empty lines or indentation'''

    qa = QA_NoContext_deepseek_R1()

    answer = qa.ask(analyze_running_error_prompt)

    advices_for_revision = answer

    return advices_for_revision

def single_file_corrector2(file_name, advices_for_revision, reference_files):
    print(f"correcting {file_name}")

    file_content = None

    file_path = f'{config.OUTPUT_PATH}/{file_name}'

    with open(file_path, "r", encoding="utf-8") as file:
        file_content = file.read()

    correct_file_prompt = f'''{config.general_prompts} Correct the OpenFOAM case file.
Please correct the { {file_name} } file with file contents as { {file_content} } to strictly adhere to the following correction advice { {advices_for_revision} }. Ensure the dimension in [] is correct if the dimension shows in the file content. You must not change any other contents of the file except for the correction advice or dimension in [].
You can reference these files from OpenFOAM tutorial { {reference_files} } for formatting.
This is the name of the boundary condition and the corresponding type [[{config.grid_info.grid_boundary_conditions}]] for this example. Please ensure that the boundary condition settings in the file comply with it.

In your final response after "Here is my response:", absolutely AVOID any elements including but not limited to:
- Markdown code block markers (``` or  ```)
- Extra comments or explanations'''

    qa = QA_NoContext_deepseek_V3()

    answer = qa.ask(correct_file_prompt)

    answer = file_writer.extract_pure_response(answer)

    try:
        file_writer.write_field_to_file(answer,file_path)
        print(f"write the file {file_name}")

    except Exception as e:
        print(f"Errors occur during write_field_to_file: {e}")
    else: # Successfully executed the field file write operation
        file_write_successful = True

def ensure_all_field_file_dimensions():

    print(f"Ensuring all field file dimensions")

    case_0_folder = f'{config.OUTPUT_PATH}/0'

    folder_path = Path(case_0_folder)

    for file_path in folder_path.iterdir():
        file_content = None
        if file_path.is_file():
            field_file = f'{case_0_folder}/{file_path.name}'

        with open(file_path, "r", encoding="utf-8") as file:
            file_content = file.read()
            a = 1

        # reference_files = find_reference_files_by_solver(field_file)

        correct_dimension_prompt =   f'''{config.general_prompts}
        Please check the dimension of the OpenFOAM field file {file_path.name} and correct it if the dimension is incorrect. The file content is: {file_content}.

        In your response: Absolutely AVOID any elements including but not limited to:
        - Markdown code block markers (``` or  ```)
        - Extra comments or explanations
        - Unnecessary empty lines or indentation
        '''

        qa = QA_NoContext_deepseek_V3()

        answer = qa.ask(correct_dimension_prompt)

        answer = file_writer.extract_pure_response(answer)

        try:
            file_writer.write_field_to_file(answer,file_path)
            print(f"write the file 0/{file_path.name}")

        except Exception as e:
            print(f"Errors occur during write_field_to_file: {e}")
        else: # Successfully executed field file write operation
            file_write_successful = True


def find_reference_files(target_file, case_name = None):
    """Find target_file files from other cases for reference
    Args:
        target_file (str): Name of the file to search for
        case_name (str or None): Name of the case currently running simulation
    Returns:
        files_content (dict): key is case_name, value is corresponding content
    """
    if case_name == None:
        case_name = config.case_name

    class ReferenceFilesContent(BaseModel):
        files: Dict[str, str] = Field(description="A dictionary where keys are case names and values are the content of the target file for that case.")
    parser = PydanticOutputParser(pydantic_object=ReferenceFilesContent)
    print("Searching for related files...")
    qa = QA_NoContext_deepseek_V3()
    select_appropriate_files = """I need to modify the {target_file} file according to the following simulation requirements. When several reference files are available, please select the {file_num} most relevant ones based on the case name and file contents.

<simulation_requirements>
{case_name}
{simulation_requirements}
</simulation_requirements>

<selectable_files>
{selectable_files}
</selectable_files>

<Output_Requirements>
1. Return the result strictly in the following JSON format:
{response_format}
2. All returned items must be chosen only from selectable_files.
3. If selectable_files lists only case names without corresponding file contents, leave the content of the {target_file} as an empty string.
</Output_Requirements>"""

    solver = config.case_solver
    turbulence_model = config.case_turbulence_model
    other_physical_model = config.other_physical_model

    # Search for reference files, try to ensure reference cases that match both solver and turbulence model
    loose = 0 # Search looseness level, 0 means consider special physical models, 1 means ignore
    file_content = {} # (case_name, file_content), [[], []]
    has_content = True # Whether the returned file_content has content
    while len(file_content) == 0 and loose < 2:
        for key, value in config.OF_case_data_dict.items():
            if solver==value["solver"] and turbulence_model == value["turbulence_model"]:
                # If there is other_physical_model, need to check if they are the same
                if loose < 1:
                    if "other_physical_model" in value.keys():
                        # print(key)
                        if other_physical_model == None or other_physical_model == []:
                            if value["other_physical_model"] != [] and value["other_physical_model"] != None and value["other_physical_model"] != ["common"]:
                                continue
                        else:
                            if not set(other_physical_model) == set(value["other_physical_model"]):  # Take intersection
                                continue
                    else:
                        loose = 1
                config_files = value['configuration_files']
                if target_file in config_files.keys():
                    file_content[key.split("/")[-1]] = config_files[target_file]
        loose += 1

    # If no cases matching both solver and turbulence model are found, consider only one of them
    if len(file_content) == 0:
        print("No cases that match both solver and turbulence model")
        file_content_sol = {}
        for key, value in config.OF_case_data_dict.items():
            if solver == value["solver"]:
                config_files = value['configuration_files']
                if target_file in config_files.keys():
                    file_content_sol[key.split("/")[-1]] = config_files[target_file]

        # If still no cases matching the solver are found, search from domain (e.g., compressible) (actually special files needed by different turbulence models)
        if len(file_content_sol) == 0:
            domain_type = None
            for key, value in config.OF_case_data_dict.items():
                if solver in key:
                    path_split = key.split('/')
                    domain_type = path_split[0]
                    break
            if domain_type is not None:
                for key, value in config.OF_case_data_dict.items():
                    path_split = key.split('/')
                    if domain_type == path_split[0]:
                        config_files = value['configuration_files']
                        if target_file in config_files.keys():
                            file_content_sol[key.split("/")[-1]] = config_files[target_file]

        # Select case files with the same solver
        if len(file_content_sol) > 2:
            # If there are more than 2 case files with the same solver, need to select 2 from them
            if len(file_content_sol) > 4:
                file_content_bak = file_content_sol.copy()

                for k,v in file_content_sol.items():
                    file_content_sol[k] = ""
                has_content = False

            file_content_sol = qa.ask(select_appropriate_files.format(
                case_name=case_name,
                target_file=target_file,
                file_num=2,
                simulation_requirements=config.case_description,
                selectable_files=file_content_sol,
                response_format=parser.get_format_instructions()
            ))

            max_retries = 3  # Maximum retry attempts
            file_content = robust_llm_parse(
                qa,
                select_appropriate_files.format(
                    case_name=case_name,
                    target_file=target_file,
                    file_num=2,
                    simulation_requirements=config.case_description,
                    selectable_files=file_content_sol,
                    response_format=parser.get_format_instructions()
                ),
                parser,
                max_retries=max_retries
            ).files

            if file_content == None:
                file_content = random.sample(list(file_content_sol.keys()), 2) if isinstance(file_content, dict) else file_content

        else:
            file_content = file_content_sol
    else:
        if len(file_content) > 2:
            if len(file_content) > 4:
                file_content_bak = file_content.copy()
                for k,v in file_content.items():
                    file_content[k] = ""
                has_content = False
            # print(file_content)
            max_retries = 3  # Maximum retry attempts
            for attempt in range(max_retries):
                try:
                    # Get file content
                    file_content = qa.ask(select_appropriate_files.format(
                        case_name=case_name,
                        target_file=target_file,
                        file_num=2,
                        simulation_requirements=config.case_description,
                        selectable_files=json.dumps(file_content, ensure_ascii=False, indent=4),
                        response_format=parser.get_format_instructions()
                    ))
                    file_content = parser.parse(file_content).files
                    break  # Parsing successful, exit loop
                except Exception as e:
                    print(f"Failed to parse when searching for reference files (attempt {attempt + 1}/{max_retries}): {e}")

                    if attempt < max_retries - 1:
                        continue  # Continue retrying
                    else:
                        print("Maximum retry attempts reached, randomly selecting 2 as reference")
                        file_content = random.sample(file_content, 2)
        else:
            file_content = json.dumps(file_content, ensure_ascii=False, indent=4)

    if not has_content:
        for k, v in file_content.items():
            file_content[k] = file_content_bak[k]
    # print("file_content:",file_content)
    return file_content

def clean_json_string(json_str):
    """Clean problematic characters in a JSON string"""
    # Remove leading and trailing whitespace
    json_str = json_str.strip()
    
    # Fix common escape character issues
    replacements = [
        (r'\\(?!["\\/bfnrtu])', r'\\\\'),
        (r'\\"', r'"'),
        (r'\\n', r'\n'),
        (r'\\t', r'\t'),
        (r'\\r', r'\r'),
    ]
    
    for old, new in replacements:
        json_str = re.sub(old, new, json_str)
    
    return json_str

def extract_content_from_response(llm_response, output_type = 'json'):
    llm_response = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', llm_response)  # 去掉非法控制字符
    if output_type == 'json' or output_type == 'dict':
        llm_response = json.dumps(llm_response)  # Handle \n and \" in values
        patterns = [
            r'```json\s*(.*?)\s*```',
            r'```\s*(.*?)\s*```',
            r'\[.*?\]'
        ]
        for pattern in patterns:
            # print(pattern)
            match = re.search(pattern, llm_response, re.DOTALL)
            if match:
                json_str = match.group(1) if '```' in pattern else match.group(0)
                # print(json_str)
                # Clean and fix JSON string
                json_str = clean_json_string(json_str)
                # print(json_str)
                try:
                    content = json.loads(json_str)
                    return content
                except json.JSONDecodeError as e:
                    # print(f"JSON parsing error: {e}")
                    continue

        # Direct processing
        try:
            content = json.loads(llm_response.strip())
        except:
            # Find the content within the first closed {} or [] in llm_response
            start = llm_response.find('{')
            start_ = llm_response.find('[')
            if start < start_:
                pos = start + 1
                depth = 1
                while pos < len(llm_response) and depth > 0:
                    if llm_response[pos] == '{':
                        depth += 1
                    elif llm_response[pos] == '}':
                        depth -= 1
                    pos += 1
            else:
                pos = start_ + 1
                depth = 1
                while pos < len(llm_response) and depth > 0:
                    if llm_response[pos] == '[':
                        depth += 1
                    elif llm_response[pos] == ']':
                        depth -= 1
                    pos += 1
            content = json.loads(llm_response[start:pos])

    elif output_type == 'str':
        pattern = r'```(.*?)```'
        match = re.search(pattern, llm_response, re.DOTALL)
        if match == None:
            content = ""
        else:
            content = match.group(1).strip()
            if content.startswith("text") or content.startswith("Text"):
                content = content[4:]
            if content.startswith("cpp"):
                content = content[3:]

    return content

def analyse_error(running_error, case_files=None, relevant_reflections = ""):
    if case_files is None:
        case_files = config.case_info.file_structure
        # print(case_files)

    class SuspiciousFilesResponse(BaseModel):
        files: Dict[str, str] = Field(description="A mapping from file name to the possible reasons for the error")

    parser = PydanticOutputParser(pydantic_object=SuspiciousFilesResponse)

    print("Searching for files that may cause OpenFOAM errors...")
    search_for_suspicious_files = f"""The following error occurred in OpenFOAM. Please identify the files that need to be modified and provide modification suggestions. The error message and output requirements are as follows:

<error_message>
{running_error}
</error_message>

<output_requirements>
1. Return the analysis results strictly in the required JSON format:
{parser.get_format_instructions()}
2. Only files listed below may contain the error:
{case_files}
</output_requirements>

<few-shot>
Analyze the error message and, in light of the physical meaning and role of each file in the list, determine which file is likely causing the error. Below are examples of how to diagnose specific errors:

1. Error: '...Entry 'hFinal' not found in dictionary "system/fvSolution/solvers"...'  
   The phrase 'in dictionary "system/fvSolution/solvers"' indicates the solver section in system/fvSolution lacks the hFinal entry. hFinal is the absolute convergence level for the final iteration when solving for h in transient algorithms. If only h is defined but hFinal is missing, the solver fails.  
   Therefore, add the hFinal keyword in the solvers subsection of system/fvSolution.

2. Error: '...Sum of mass fractions is zero for species...'  
   This occurs when the sum of all species mass fractions does not equal 1. Check every species (e.g., CH4, CO2) in each medium (e.g., fuel, air) and ensure their mass fractions sum to 1.  
   Files to inspect: 0/CH4, 0/CO2, etc., and correct any inconsistent values.

3. Error: '...#0 Foam::error::printStack(Foam::Ostream&) at ??:?...Floating point exception'  
   A floating-point exception usually stems from:  
   1) Extremely large or small numbers, or physically invalid zeros/negatives;  
   2) Misunderstood dimensions (e.g., for incompressible flow p has dimensions [0 2 -2 0 0 0 0] and can be 0 Pa, whereas for compressible flow p has dimensions [1 -1 -2 0 0 0 0] and must include atmospheric pressure, e.g., 1e5 Pa);  
   3) Incorrect boundary conditions in field files.  
   Inspect relevant field files and property files, such as 0/p and constant/transportProperties.

4. Error: '...Entry 'specie' not found in dictionary "constant/thermo.compressibleGas/mixture"...' with file list containing '0/CH4, constant/thermophysicalProperties, constant/thermo.compressibleGas'  
   The error arises because 'specie' is missing in the mixture sub-dictionary of constant/thermo.compressibleGas; however, 'mixture' belongs in constant/thermophysicalProperties, not in constant/thermo.compressibleGas.  
   The correct fix is to remove the 'mixture' keyword from constant/thermo.compressibleGas and add it in constant/thermophysicalProperties where the thermophysical model is defined.
</few-shot>"""

    if relevant_reflections != "":
        search_for_suspicious_files += f"\n{relevant_reflections}"
    qa = QA_NoContext_deepseek_V3()

    suspicious_files = None

    response = qa.ask(search_for_suspicious_files)
    suspicious_files = parser.parse(response).files
    print("suspicious_files:\n", suspicious_files)

    if len(suspicious_files) > 1:   # If suspicious files > 1, need to check combined with file content

        # Get content from suspicious files
        suspicious_file_content = {}
        for name, reason in suspicious_files.items():
            if os.path.exists(os.path.join(config.path_cfg.output_case_path, name)) != False:
                with open(os.path.join(config.path_cfg.output_case_path, name), "r") as f:
                    file_content = f.read()
                    suspicious_file_content[name] = file_content

        class ErrorFilesResponse(BaseModel):
            files: Dict[str, str] = Field(description="A mapping from file name to the reasons for the error")

        parser = PydanticOutputParser(pydantic_object=ErrorFilesResponse)

        print("Confirming files that cause OpenFOAM errors...")
        search_for_error_files = f"""OpenFOAM reported an error. After preliminary analysis, the files below are suspected. I will provide the exact error message and the contents of these files; please identify the true culprit.

<error_message>
{running_error}
</error_message>

<Suspicious_file>
{suspicious_files}
</Suspicious_file>

<file_content>
{suspicious_file_content}
</file_content>

<Output_Requirements>
1. Return the analysis result strictly in the requested JSON format:
{parser.get_format_instructions()}
2. Only files listed in Suspicious_file may be considered as the source of the error
</Output_Requirements>

<few-shot>
Interpret the error message in light of each file's content, physical meaning, and role to locate the erroneous file. Examples below illustrate the thought process for common errors:

1. Error: "...Entry 'hFinal' not found in dictionary 'system/fvSolution/solvers'..."
   The missing entry triggers the error. Inspecting system/fvSolution reveals that 'h' is defined but 'hFinal' is absent. 'hFinal' denotes the absolute convergence level for the final iteration when solving for h in transient algorithms. Therefore, add the 'hFinal' definition inside the solvers subsection of system/fvSolution.

2. Error: "...Sum of mass fractions is zero for species..."
   This occurs when the sum of mass fractions for all species does not equal 1. Files 0/CH4 and 0/CO2 are present. Their fractions sum to 1 in the 'air' region but to 2 in the 'fuel' region. Adjust the mass fractions in 0/CH4 and 0/CO2 under 'fuel' so that their sum is exactly 1.

3. Error: "...#0 Foam::error::printStack(Foam::Ostream&) at ??:?...Floating point exception"
   A floating-point exception typically arises from: 1) physically invalid or extreme values, 2) misunderstandings (e.g., setting p=0 for incompressible flow is acceptable as relative pressure, but for compressible flow p must include atmospheric pressure, e.g., 1e5 Pa), 3) improper boundary conditions in field files.
   Upon checking 0/p, constant/transportProperties, and system/fvSolution, the internal pressure is 0, which is valid for incompressible flow, so 0/p is not the issue. However, system/fvSolution sets the relaxation factor for p to 1.5; values greater than 1 can destabilize the solution. Reduce the relaxation factor for p in system/fvSolution.

4. Error: "...Entry 'specie' not found in dictionary 'constant/thermo.compressibleGas/mixture'...", with files 0/CH4, constant/thermophysicalProperties, constant/thermo.compressibleGas.
   The superficial cause is a missing 'specie' entry inside the 'mixture' section of constant/thermo.compressibleGas. However, constant/thermo.compressibleGas defines pure-gas properties, whereas 'mixture' belongs in the thermoType section of constant/thermophysicalProperties. The actual issue is the misplaced 'mixture' keyword.
   Observing constant/thermo.compressibleGas shows gas properties nested under 'mixture', yet pure gases (e.g., O2, CH4) must appear as top-level entries. Remove the 'mixture' keyword from constant/thermo.compressibleGas and list the gas properties directly at the top level. constant/thermophysicalProperties already has 'mixture' correctly placed under thermoType and requires no further changes.
</few-shot>"""

        if relevant_reflections != "":
            search_for_error_files += f"\n{relevant_reflections}"

        qa = QA_NoContext_deepseek_R1()

        error_files = None
        max_retries = 3  # Set maximum retry count
        for attempt in range(max_retries):
            try:
                response = qa.ask(search_for_error_files)
                error_files = parser.parse(response).files
                break  # If parsing is successful, exit the loop
            except Exception as e:
                print(f"Parsing failed (attempt {attempt + 1}/{max_retries}): {e}")
                if attempt < max_retries - 1:  # If not the last attempt
                    print("Retrying...")
                    continue
                else:
                    print("Reached maximum retry count, unable to parse response, using suspicious files directly")
                    error_files = suspicious_files   # If
                    break

        print("error_files:\n", suspicious_files)

    else:
        error_files = suspicious_files

    return error_files

def correct_error(running_error, error_files, case_files=None, relevant_reflections = ""):
    """Modify error files to solve OpenFOAM runtime errors
    Args:
        running_error(str): OpenFOAM runtime error information
        error_files(dict): Files that may cause errors and their corresponding error reasons
        case_files(list or str): List of case file structure
    Returns:
        files_corrected(list): List of modified files
    """
    if case_files is None:
        case_files = config.case_info.file_structure

    class FileReference(BaseModel):
        """FileReference={'reference_files':[], 'reference_reason':''}"""
        reference_files: List[str] = Field(description="List of files to be referred to during the revision process")
        reference_reason: str = Field(description="The reason for referring to these documents")
    class ErrorFilesResponse(BaseModel):
        """{'error_files':{'': FileReference}}"""
        error_files: Dict[str, FileReference] = Field(description="Error files and their reference information")
    parser = PydanticOutputParser(pydantic_object=ErrorFilesResponse)   # Create parser

    search_relevant_files = f'''The OpenFOAM case reports the following error. After analysis, the listed files are suspected. When fixing these files, cross-file references may be necessary. Please list, for each file to be modified, the other files that should be consulted and the reasons why.

<error_message>
{running_error}
</error_message>

<error_files>
{error_files}
</error_files>

<Output_Requirements>
1. Return the result strictly in the following format; all fields must be filled:
{parser.get_format_instructions()}
2. reference_files must be an array; if no other files need to be referenced, return an empty array []
3. Only files in the list below may be used as references:
{case_files}
</Output_Requirements>

<few-shot>
Combine the error message with the content, physical meaning, and role of each file in the list to deduce inter-file dependencies. The following examples illustrate the thought process:
1. Error: "...Entry 'hFinal' not found in dictionary 'system/fvSolution/solvers'..."  
   Suspect file: system/fvSolution  
   File list includes: 0/U, 0/p, system/fvSolution…  
   The error occurs because hFinal is missing in the solvers subsection of system/fvSolution. hFinal is the final absolute convergence level for the transient solver of h.  
   Since hFinal is unrelated to any other file, no cross-reference is needed. Return:
   {{"system/fvSolution": {{"reference_files": [],"reference_reason": "The error is caused by the absence of the hFinal entry in system/fvSolution. Adding hFinal is self-contained; no coupling with other files exists."}}}}
2. Error: "...Sum of mass fractions is zero for species..."  
   Suspect files: 0/CH4, 0/CO2  
   File list includes: 0/U, 0/CH4, 0/CO2, constant/turbulenceProperties…  
   The error arises because the sum of species mass fractions does not equal 1. One must check the mass fractions of CH4 and CO2 in each region (e.g., fuel, air) and ensure their sum is exactly 1.  
   Files 0/CH4 and 0/CO2 store the respective species fractions. To enforce the unity sum, each file must reference the other. Return:
   {{"0/CH4": {{"reference_files": ["0/CO2"],"reference_reason": "0/CH4 and 0/CO2 define species mass fractions. To ensure the fractions sum to 1, the CH4 file must consult the CO2 file to obtain the complementary fraction."}},"0/CO2": {{"reference_files": ["0/CH4"],"reference_reason": "0/CH4 and 0/CO2 define species mass fractions. To ensure the fractions sum to 1, the CO2 file must consult the CH4 file to obtain the complementary fraction."}}}}
</few-shot>'''

    print("Detecting which files in the case need additional reference when modifying error files...")
    qa = QA_NoContext_deepseek_V3()

    files_need_to_correct = None

    response = qa.ask(search_relevant_files)
    try:
        # Parse the response using the Pydantic parser
        files_need_to_correct = parser.parse(response).model_dump()
        # return files_need_to_correct
    except Exception as e:
        print(f"Parsing failed: {e}")
        # return None

    error_files_names = []  # Names of files that may have errors
    relevant_files = {}     # key: names of files that may have errors, value: names of related files
    reference_files = {}    # key: names of files that may have errors, value: content of that file in tutorial cases
    for k, v in files_need_to_correct['error_files'].items():
        k_clean = k.strip().strip('"').strip("'").strip()
        for v_ in v["reference_files"]:
            if v_ not in case_files:
                continue
        error_files_names.append(k_clean)
        relevant_files[k_clean] = v["reference_files"]
        reference_files[k_clean] = find_reference_files(k_clean)
        
    files_content = {}     # key: names of files used for modification and reference, value: corresponding file content
    for k, v in relevant_files.items():
        if k not in files_content.keys():
            if os.path.exists(os.path.join(config.path_cfg.output_case_path, k)) != False:
                with open(os.path.join(config.path_cfg.output_case_path, k), "r") as f:
                    files_content[k] = f.read()
        for v_ in v:
            if v_ not in files_content.keys():
                if os.path.exists(os.path.join(config.path_cfg.output_case_path, v_)) != False:
                    with open(os.path.join(config.path_cfg.output_case_path, v_), "r") as f:
                        files_content[v_] = f.read()

    processed_files = set()
    correcting_advice = {}
    for error_file_name in error_files_names:

        # List files related to error_file_name that may need to be changed later
        special_files = (set(relevant_files[error_file_name])&set(error_files_names)) - processed_files
        processed_files.add(error_file_name)

        if special_files:
            tips = f"There is a small possibility that the OpenFOAM error is related to {special_files}; modifications to {special_files} will be addressed later."
            output_requirements = f"""1) If the file is correct, simply return NO without any additional content.
2) {tips}
3) Do not violate the case requirements unless the settings are clearly unreasonable and directly cause the error.
4) Provide detailed, concrete modification steps for {error_file_name}; do not include reasoning or explanations."""
        else:
            output_requirements = f"""1) If the file is correct, simply return NO without any additional content.
2) Do not violate the case requirements unless the settings are clearly unreasonable and directly cause the error.
3) Provide detailed, concrete modification steps for {error_file_name}; do not include reasoning or explanations."""
            
        relevant_files_content = {}
        for ref_file in relevant_files[error_file_name]:
            if ref_file in files_content.keys():
                relevant_files_content[ref_file] = files_content[ref_file]

        advice_based_on_ref = f"""OpenFOAM reported an error that is likely caused by the file '{error_file_name}'. Please analyze the cause based on the information below and provide detailed, concrete suggestions for correcting '{error_file_name}':

1. OpenFOAM error message:
<error_message>
{running_error}
</error_message>

2. Content of '{error_file_name}':
<{error_file_name}_content>
{files_content[error_file_name]}
</{error_file_name}_content>

3. Possible reason for the error in '{error_file_name}':
<error_reason>
{error_files[error_file_name]}
</error_reason>

4. Content of files related to '{error_file_name}' in this case:
<relevant_files>
{relevant_files_content}
</relevant_files>

5. Settings for '{error_file_name}' in other OpenFOAM tutorial cases:
<{error_file_name}_of_other_case>
{reference_files[error_file_name]}
</{error_file_name}_of_other_case>

6. Case configuration requirements:
<case_requirements>
{config.case_description}
</case_requirements>

Output requirements:
{output_requirements}"""
        if relevant_reflections != "":
            advice_based_on_ref += f"\n{relevant_reflections}"

        MAX_LENGTH = 98304/2  # The maximum input length limit of OpenAI
        if len(advice_based_on_ref) > MAX_LENGTH:
            print(advice_based_on_ref)
            advice_based_on_ref = advice_based_on_ref[:MAX_LENGTH] + '... ...'

        print("Proposing modification suggestions for error files...")
        qa = QA_NoContext_deepseek_R1()
        response = qa.ask(advice_based_on_ref)
        if "NO" not in response[:30]:
            correcting_advice[error_file_name] = response
    
    # files_corrected = {}
    files_corrected = {}
    print("Modifying error files according to suggestions...")
    qa = QA_NoContext_deepseek_V3()
    for error_file_name, advice in correcting_advice.items():

        correct_error_file = f"""OpenFOAM encountered an error that is likely caused by the file '{error_file_name}'. Please analyze the cause based on the information below, correct the file, and return the result in the specified format:

1. OpenFOAM error message:
    <error_message>
    {running_error}
    </error_message>

2. Content of '{error_file_name}':
    <{error_file_name}_content>
    {files_content[error_file_name]}
    </{error_file_name}_content>

3. Revision advice:
    <revision_advice>
    {advice}
    </revision_advice>

4. Settings for '{error_file_name}' in other OpenFOAM tutorial cases:
    <{error_file_name}_of_other_case>
    {reference_files}
    </{error_file_name}_of_other_case>

5. Mesh boundary conditions:
    <mesh_boundary_condition>
    {config.grid_info.grid_boundary_conditions}
    </mesh_boundary_condition>

6. Case configuration requirements:
    <case_requirements>
    {config.case_description}
    </case_requirements>

Output requirements:
1) If '{error_file_name}' is actually correct, return NO; otherwise, return the fully corrected content.
2) Place the returned content between ``` and ```, with no additional text.
3) Do not violate the case configuration requirements unless the settings are clearly unreasonable and directly cause the error.
4) When setting boundary conditions for physical fields, take the mesh boundary conditions into account to avoid conflicts."""

        response = qa.ask(correct_error_file)
        if "NO" not in response:

            with open(os.path.join(config.path_cfg.output_case_path, error_file_name), "w") as f:
                print(f"Modified: {os.path.join(config.path_cfg.output_case_path, error_file_name)}")
                new_file_content = extract_content_from_response(response,"str")
                f.write(new_file_content)
                
            files_corrected[error_file_name] = [files_content[error_file_name], new_file_content]
    print(f"Modified: {files_corrected.keys()}")
    return files_corrected
