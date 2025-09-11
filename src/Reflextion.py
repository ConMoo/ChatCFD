import json
import os
from datetime import datetime
from qa_modules import QA_NoContext_deepseek_V3, QA_NoContext_deepseek_R1
import config


# Global reflection record storage
reflection_history = [] # Elements are dict, key is error message, value is reflection

def reflextion(running_error, correct_trajectory):
    """Reflect on the reasons for error correction failure and save experience data
    Args:
        running_error (str): Runtime error information
        correct_trajectory (list): Modification trajectory, elements as {file_name: [original_content, modified_content], }    
    Returns:
        str: Reflection result
    """
    global reflection_history
    
    try:
        # Generate reflection prompt
        reflection_prompt = generate_reflection_prompt(running_error, correct_trajectory)
        
        # Call large model for reflection
        qa = QA_NoContext_deepseek_R1()
        reflection_result = qa.ask(reflection_prompt)
        
        # Build reflection data record and add to global reflection history
        reflection_record = {
            "running_error": running_error,
            "reflection_result": reflection_result,
        }
        reflection_history.append(reflection_record)
        
        # # Save to file
        # save_reflection_to_file(reflection_record)
        # print(f"Reflection completed, saved to file. Current reflection record count: {len(reflection_history)}")
        
        return reflection_result
        
    except Exception as e:
        print(f"Error occurred during reflection process: {e}")
        return f""

def generate_reflection_prompt(running_error, correct_trajectory):
    """Generate reflection prompt"""
    
    # Build file comparison information
    file_comparison = ""
    for idx, trial in enumerate(correct_trajectory):
        file_comparison = f"### trial_{idx}:\n"
        # for 
        for file_name, file_change in trial.items():    
            if len(file_change) > 1:
                file_comparison += f"\nModified file: {file_name} ===\n"
                file_comparison += f"Original content:\n{file_change[0]}\n"
                file_comparison += f"Modified content:\n{file_change[1]}\n"
            else:
                file_comparison += f"\n=== New file: {file_name} ===\n"
                file_comparison += f"Content:\n{file_change[0]}\n"

    reflection_prompt = f"""When running the {config.case_name} case using OpenFOAM, the following error occurred:
<error_information>
{running_error}
</error_information>

You have attempted the following modifications to resolve the issue:
<file_modifications>
{file_comparison}
</file_modifications>

However, the error still persists. Please reflect on the reasons for the task failure, identify the error file causing the error, point out the content that should be checked, consider previous oversights, and describe the actions to be taken next. Example: I tried A and B but forgot C. Next, I will take action D to solve the problem.
The error could only be caused by incorrect content in the following files:
<files_list>
{config.case_info.file_structure}
</files_list>

Please state your reflection content after the title "Reflection:".

Here are two examples of reflections:
<examples>
1. When simulating the counterFlowFlame2D case, an error "File 'reactions' does not exist" occurred. I generated a new reactions file according to the OpenFOAM tutorial and confirmed the path and format, but I neglected to check if there is a line "foamChemistryFile "<constant>/reactions"" in thermophysicalProperties to point to the reactions file. Therefore, when the error "File 'reactions' does not exist" occurs, it is necessary to check thermophysicalProperties to ensure the existence of this pointing line.
2. When running the BuoyantCavity case, an error "Initial temperature T0 is negative" occurred. I checked the settings of 0/T, 0/p, constant/thermophysicalProperties, system/fvSolution, and system/fvSchemes, but I overlooked checking whether the boundary conditions of p_rgh are reasonable. Thus, after checking 0/T, 0/p, etc., if the error "Initial temperature T0 is negative" still exists, I will continue to set appropriate internal field values and boundary conditions for p_rgh and related files.
</examples>

Reflection:"""

    return reflection_prompt

def construct_reflection_context(running_error, reflection_history):
    context = []
    context_other = []

    for record in reflection_history:
        if record['running_error'] == running_error:
            context.append(record)
        else:
            context_other.append(record)

    if context != []:
        if len(context) > 3:
            context = context[-3:]  # Keep at most 3
        if len(context) < 3:
            context.extend(context_other[-(3-len(context)):])  # Fill up to 3

        relevant_reflections = "<reflection>\nYou were unable to successfully complete this task before. Below are your reflections. Please refer to these reflections to avoid repeating errors:"
        for idx, record in enumerate(context):
            reflection = record['reflection_result']
            relevant_reflections += f"\n### reflection_{idx}\n{reflection}"

        relevant_reflections += "\n</reflection>"
    else:
        relevant_reflections = ""
    return relevant_reflections

