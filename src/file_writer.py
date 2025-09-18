import config
import shutil
import os

from qa_modules import QA_NoContext_deepseek_V3,QA_NoContext_deepseek_R1
from PyFoam.RunDictionary.ParsedParameterFile import ParsedParameterFile

import json
import re

def extract_content_in_brackets(text, indicator_string):
    # double brackets
    pattern = fr'{indicator_string} \[\[(.*?)\]\]'
    matches = re.findall(pattern, text, re.DOTALL)
    return [m.strip() for m in matches]

def extract_foamfile_content(input_string, indicator_string):
    start_tag = r"\\Start_" + indicator_string
    end_tag = r"\\End_" + indicator_string
    
    # Use regex to find all matches
    pattern = re.compile(f"{start_tag}(.*?){end_tag}", re.DOTALL)
    matches = pattern.findall(input_string)
    
    # Filter out content containing "FoamFile"
    foamfile_matches = [match for match in matches if 'FoamFile' in match]
    
    if len(foamfile_matches) == 0:
        return "No content containing 'FoamFile' found"
    elif len(foamfile_matches) > 1:
        return "Error: Multiple contents containing 'FoamFile' found"
    else:
        return foamfile_matches[0]
    
def extract_pure_response(text):
    # Use regex to match all content (including newlines)
    pattern = r"Here is my response:(.*?)(?=$|\Z)"
    match = re.search(pattern, text, re.DOTALL)
    
    if match:
        # Remove leading and trailing whitespace
        return match.group(1).strip()
    return ""

def remove_functions_blocks(text):
    pattern = r'functions\s*\{.*?\}'
    return re.sub(pattern, '', text, flags=re.DOTALL)

def write_field_to_file(field_file_content, output_file_name):
    # Escape processing (handle \n and special symbols)
    processed_content = field_file_content.encode('latin-1').decode('unicode_escape')

    directory = os.path.dirname(output_file_name)

    if directory and not os.path.exists(directory):
        os.makedirs(directory)

    # Write to file (recommended to use the filename corresponding to the object field)
    with open(output_file_name, 'w', encoding='utf-8') as f:  # Filename suggested to use "U" based on object field
        f.write(processed_content)

# 1. Determine which constant files to write based on physical model and solver;
# 2. Write files by referencing the corresponding constant files
def copy_folder(source_folder, destination_folder):
    """
    Copy a folder and its contents from source_folder to destination_folder with specific exclusions.

    :param source_folder: Path to the source folder.
    :param destination_folder: Path to where the folder should be copied.
    :return: None
    """
    # If the destination folder does not exist, create it
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)

    destination_path = os.path.join(destination_folder, os.path.basename(source_folder))

    try:
        for root, dirs, files in os.walk(source_folder):
            # Relative path, used to build the same structure in the destination folder
            rel_path = os.path.relpath(root, source_folder)
            for dir in dirs:
                if os.path.join(rel_path, dir).startswith('constant/polyMesh'):
                    dirs.remove(dir)  # Do not copy folders under constant/polyMesh
            new_root = os.path.join(destination_path, rel_path)
            os.makedirs(new_root, exist_ok=True)
            
            for file in files:
                source_file = os.path.join(root, file)
                dest_file = os.path.join(new_root, file)
                
                # Check if the file size exceeds 25k (25 * 1024 bytes)
                if os.path.getsize(source_file) > 25 * 1024:
                    print(f"Skipping file {source_file} because its size exceeds 25k")
                    continue
                
                # Do not copy .msh files
                if file.endswith('.msh'):
                    print(f"Skipping file {source_file} because it is a .msh file")
                    continue

                shutil.copy2(source_file, dest_file)
                print(f"Successfully copied file {source_file} to {dest_file}")

        print(f"Successfully copied {source_folder} to {destination_folder}")
    except shutil.Error as e:
        print(f"Error copying folder: {e}")
    except OSError as e:
        print(f"Error creating directory or copying file: {e.strerror}")

def analyze_running_error(running_error):

    str_global_files = ", ".join(config.global_files)

    analyze_running_error_prompt = f'''{config.general_prompts}
    Examine the following error encountered while running an OpenFOAM case, and determine which file needs revision to correct the error. The candidate files are {str_global_files}.
    - Indicate the file requiring revision with File_for_revision [[file_name]], where file_name within the double brackets is the exact file needing changes.
    - Provide your recommendations for correcting the file with Advice_for_revision [[actual_advices]], where actual_advices within the double brackets are specific suggestions to fix the error
    
    The error is {running_error}.
    '''
    
    async_qa = QA_NoContext_deepseek_V3()

    answer = async_qa.ask(analyze_running_error_prompt)

    pure_response = extract_pure_response(answer)

    file_for_revision = extract_content_in_brackets(pure_response, "File_for_revision")[0]

    advices_for_revision = extract_content_in_brackets(pure_response, "Advice_for_revision")[0]

    config.global_file_requirement[file_for_revision]['correction_advice'] = advices_for_revision

    return file_for_revision