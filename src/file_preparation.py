import os
import json
import re

import subprocess
from PyFoam.RunDictionary.ParsedParameterFile import ParsedParameterFile
from PyFoam.Basics.DataStructures import BinaryList

import config
import pdf_chunk_ask_question
from qa_modules import QA_NoContext_deepseek_V3,QA_NoContext_deepseek_R1
from file_corrector import extract_content_from_response, find_reference_files

from pydantic import BaseModel, Field
from typing import List, Dict
from langchain.output_parsers import PydanticOutputParser


def convert_mesh(output_case_path=None, grid_path=None, grid_type=None):
    """Convert .msh mesh to OpenFOAM format, or copy polyMesh mesh to OpenFOAM case
    Args:
        case_path (str): OpenFOAM case path (output_path)
        grid_path (str): Mesh file path
    Returns:
        bool: Whether conversion/copying is successful
    """
    if output_case_path is None:
        output_case_path = config.path_cfg.output_case_path
    if grid_path is None:
        grid_path = config.path_cfg.grid_path
    if grid_type is None:
        grid_type = config.grid_type
    try:
        constant_path = os.path.join(output_case_path, "constant")

        if grid_type == "msh":
            command = [
                "fluentMeshToFoam",
                "-case",
                output_case_path,
                grid_path
            ]
            subprocess.run(command, check=True)
            print("Mesh conversion completed successfully")
        elif grid_type == "polyMesh":
            if not os.path.exists(constant_path):
                os.makedirs(constant_path)
            print(f"Copying mesh to {constant_path}/")
            command = f"cp -r {grid_path} {constant_path}"
            # print(command)

            subprocess.run(command, shell=True, check=True)
            print("Mesh loaded successfully")

        config.mesh_convert_success = True
        # Save initial boundary file
        boundary_path = os.path.join(constant_path, "polyMesh/boundary")
        with open(boundary_path, "r", encoding="utf-8") as file:
            boundary_content = file.read()
        config.grid_info.grid_boundary_init = boundary_content

        return True
    except subprocess.CalledProcessError as e:
        print(f"Mesh conversion failed: {e}")
        return False
    except FileNotFoundError:
        import traceback
        traceback.print_exc()
        print("Related files or directories not found")
        return False

def extract_boundary_names(grid_file_path, grid_type = None):
    """
    Extract boundary condition names from fluent.msh mesh file
    Args:
        grid_file_path (str): Mesh file path to process
        grid_type (str): Mesh type, supports "msh" and "polyMesh"
    Returns:
        result list (list): List of extracted boundary condition names (dict for polyMesh)
    """
    if grid_file_path == None:
        grid_file_path = config.path_cfg.grid_path
    if grid_type == None:
        grid_type = config.grid_type
    if grid_type == "msh":
        # Read file content
        with open(grid_file_path, 'r') as f:
            content = f.read().splitlines()

        # Find starting point
        start_index = -1
        for i in range(len(content)-1, -1, -1):  # Find starting line from back to front
            if content[i].strip() == '(0 "Zone Sections")':
                start_index = i
                break

            if content[i].strip().startswith('4 4 4 4 4 4 4 4 4'):
                start_index = i
                break
        
        if start_index == -1:
            return []

        # Extract relevant lines
        pattern = re.compile(r'\(\d+\s+\(\d+\s+\S+\s+(\S+)\)\(\)\)')
        results = []
        
        # Process each line of data
        for line in content[start_index+1:]:
            line = line.strip()
            if not line.startswith('(39'):
                continue
                
            match = pattern.match(line)
            if match:
                value = match.group(1)
                # Filter *_FLUID and *_SOLID
                if not re.search(r'^(FLUID|\w+?_FLUID|\w+?_SOLID)$', value):
                    results.append(value)
        
        # config.case_boundaries = results
        return results
    elif grid_type == "polyMesh":
        boundary_content = ""
        boundary_names = []
        boundary_types = []
    
        try:
            boundary_content = ParsedParameterFile(os.path.join(grid_file_path, "boundary"), boundaryDict=True).content
            if type(boundary_content) == BinaryList:
                raise ValueError("Binary content is not supported")
            numOfBoundaries = len(boundary_content) // 2    # 0,2,4... are boundary names, 1,3,5... are boundary types
            for i in range(numOfBoundaries):
                boundary_names.append(boundary_content[i * 2])
                boundary_types.append(boundary_content[i * 2 + 1]["type"])

            # print("boundary_names:\n", boundary_names)
            # print("boundary_types:\n", boundary_types)
            if len(boundary_names) != len(boundary_types):
                 print("Boundary names and types count mismatch")
                 raise ValueError("Boundary names and types count mismatch")
            else:
                # Combine boundary names and types into dictionary
                boundary_dict = dict(zip(boundary_names, boundary_types))
                return boundary_dict  # .........
                # return boundary_names
        except:
            if type(boundary_content) == BinaryList:
                boundary_content = boundary_content.data
            else:
                with open(os.path.join(grid_file_path, "boundary"), 'r', encoding='utf-8') as file:
                    boundary_content = file.read()

            pattern = r'(\w+)\s*\{[^}]*?type\s+(\w+);'
            # pattern = r'(\w+)\s*\n\s*\{\s*type\s+(\w+);'
            matches = re.findall(pattern, boundary_content, re.DOTALL)
            boundary_dict = {name: b_type for name, b_type in matches}
            return boundary_dict 
    else:        
        print("Only supports msh and polyMesh format mesh files")
        return []

def setup_cfl_control(case_path, max_co=0.6, controlDict_ref=None):
    """Set CFL control parameters"""
    demo_compressible_solver = ["rhoCentralFoam", "sonicFoam"]
    if controlDict_ref is None or solver in demo_compressible_solver:
        try:
            # Modify controlDict file
            control_dict_path = f'{case_path}/system/controlDict'
            control_dict = ParsedParameterFile(control_dict_path)

            # demo_compressible_solver = ["rhoCentralFoam", "sonicFoam"]

            solver = control_dict["application"]
            if solver in config.steady_solvers:
                control_dict["adjustTimeStep"] = "yes"
                control_dict["maxCo"] = max_co
                control_dict["startTime"] = 0
                control_dict["endTime"] = 10
                control_dict["stopAt"] = "endTime"
                control_dict["writeControl"] = "timeStep"
                control_dict["writeInterval"] = 5
                control_dict["deltaT"] = 1
                control_dict["purgeWrite"] = 20    # Keep only the latest 20 time steps
                control_dict["minDeltaT"] = 1e-10   # Set minimum time step
                control_dict["maxAlphaCo"] = max_co
            else:
                control_dict["adjustTimeStep"] = "yes"
                control_dict["maxCo"] = max_co
                control_dict["maxAlphaCo"] = max_co
                control_dict["startTime"] = 0
                dt = 1e-8
                if solver in demo_compressible_solver:
                    dt = 1e-8
                else:
                    dt = 1e-5
                control_dict["deltaT"] = dt
                control_dict["endTime"] = dt*10
                control_dict["stopAt"] = "endTime"
                control_dict["writeControl"] = "timeStep"
                control_dict["writeInterval"] = 2
                if solver in demo_compressible_solver:
                    control_dict["deltaT"] = 1e-8
                else:
                    control_dict["deltaT"] = 1e-5
                control_dict["purgeWrite"] = 20    # Keep only the latest 20 time steps
                control_dict["minDeltaT"] = 1e-10   # Set minimum time step
            
            # Save modifications
            control_dict.writeFile()
            config.set_controlDict_time = True
            print("Successfully configured CFL control parameters")
            return True
        except Exception as e:
            print(f"Failed to modify controlDict: {e}")
            raise e
    else:
        
        class deltaTSetting(BaseModel):
            """{'deltaT':str}"""
            deltaT: str = Field(description="The value of deltaT")

        parser = PydanticOutputParser(pydantic_object=deltaTSetting)   # Create parser
        qa = QA_NoContext_deepseek_V3()
        set_deltaT_prompt = f"""I will simulate the following case using OpenFOAM-v2406 and need to set deltaT in controlDict. Please help me complete this setting:

<case_requirements>
- Solver: {config.case_info.case_solver}
- Turbulence model: {config.case_info.turbulence_model}
- Case description: {config.case_info.case_description}
</case_requirements>

Here is a controlDict file from the OpenFOAM tutorials for your reference:

<reference_files>
{controlDict_ref}
</reference_files>

Please strictly follow the requirements below for the output:

<output_requirements>
1. Return the result exactly in the format specified below:
{parser.get_format_instructions()}
2. Do not include any additional text, explanations, or Markdown formatting.
</output_requirements>"""

        deltaT_value = parser.parse(qa.ask(set_deltaT_prompt)).deltaT
        # print(f"deltaT_value: {deltaT_value}")

        try:
            # Modify controlDict file
            control_dict_path = f'{case_path}/system/controlDict'
            control_dict = ParsedParameterFile(control_dict_path)

            control_dict["startTime"] = 0
            control_dict["endTime"] = float(deltaT_value)*10
            control_dict["stopAt"] = "endTime"
            control_dict["writeControl"] = "timeStep"
            control_dict["writeInterval"] = 2
            control_dict["deltaT"] = deltaT_value
            control_dict["purgeWrite"] = 20    # Keep only the latest 20 time steps
            control_dict["minDeltaT"] = 1e-10   # Set minimum time step
            control_dict["adjustTimeStep"] = "yes"
            control_dict["maxCo"] = max_co
            control_dict["maxAlphaCo"] = max_co

            # Save modifications
            control_dict.writeFile()
            config.set_controlDict_time = True
            print("Successfully configured CFL control parameters")
            return True
        except Exception as e:
            print(f"Failed to modify controlDict: {e}")
            raise e

def write_field_to_file(field_file_content, output_file_name):
    # Escape processing (handle \n and special characters)
    processed_content = field_file_content.encode('latin-1').decode('unicode_escape')

    directory = os.path.dirname(output_file_name)

    if directory and not os.path.exists(directory):
        os.makedirs(directory)

    # Write to file (recommended to use filename corresponding to object field)
    with open(output_file_name, 'w', encoding='utf-8') as f:  # Filename should be based on object field, recommended to use "U"
        f.write(processed_content)


# Select file composition
def case_required_files(solver=None, turbulence_model=None, other_physical_model=None, case_name=None):
    """Generate required file list based on solver, turbulence model and other information
    Args:
        solver: Case solver
        turbulence_model: Case turbulence model
        case_name: Case name
    Returns:
        required_files (set): List of files to be generated
        file_turbulence_model(dict): Reference cases for generating file structure and their corresponding turbulence models
    """
    # if solver == None:
    #     solver=config.case_solver
    config.case_solver = solver
    config.case_turbulence_model = turbulence_model
    # if turbulence_model == None:
    #     turbulence_model=config.case_turbulence_model
    if other_physical_model == None:
        other_physical_model = config.case_info.other_physical_model
    if case_name == None:
        case_name = config.case_info.other_physical_model

    # Find reference files and select required files for the case based on simulation requirements
    file_alternative = {}
    system_necessary = ["system/fvSolution","system/controlDict","system/fvSchemes","system/FOBilgerMixtureFraction", "system/fvOptions", "system/setFieldsDict", "system/setAlphaFieldDict", "system/mapFieldsDict","system/faOptions",
                        "system/finite-area/faSchemes", "system/finite-area/faSolution", "system/finite-area/faMeshDefinition"
                        "system/optimisationDict", "system/FOXiReactionRate", "system/momentum"]

    # Ensure the solver and turbulence model are the same, and try to keep other physical models the same as much as possible
    loose = 0 # Search looseness level, 0 means considering special physical models, 1 means not considering
    while len(file_alternative) == 0 and loose < 2:
        for key, value in config.OF_case_data_dict.items():
            # Ensure the solver is the same
            if solver==value["solver"] and turbulence_model == value["turbulence_model"]:
                if loose < 1:
                    if "other_physical_model" in value.keys():
                        if other_physical_model == None or other_physical_model == []:
                            if value["other_physical_model"] != [] and value["other_physical_model"] != None and value["other_physical_model"] != ["common"]:
                                continue  
                        else:
                            if not set(other_physical_model) == set(value["other_physical_model"]):
                                continue
                    else:
                        loose = 1
                file_alternative[key.split("/")[-1]] = set(value['configuration_files'].keys())
        loose += 1

    file_turbulence_model = {}
    if len(file_alternative) == 0:
        # Ensure the solver must be the same
        loose = 0
        while len(file_alternative) == 0 and loose < 2:
            for key, value in config.OF_case_data_dict.items():
                if solver==value["solver"]:
                    if loose < 1:
                        if "other_physical_model" in value.keys():
                            if other_physical_model == None or other_physical_model == []:
                                if value["other_physical_model"] != [] and value["other_physical_model"] != None and value["other_physical_model"] != ["common"]:
                                    continue  
                            else:
                                if not set(other_physical_model) == set(value["other_physical_model"]):
                                    continue
                        else:
                            loose = 1
                    file_alternative[key.split("/")[-1]] = set(value['configuration_files'].keys())
                    file_turbulence_model[key.split("/")[-1]] = value["turbulence_model"]
            loose += 1
    # print(file_turbulence_model)

    if len(file_alternative) == 0:
        # If not found, search from domain
        domain_type = None
        for key, value in config.OF_case_data_dict.items():
            if solver in key:
                path_split = key.split('/')
                domain_type = path_split[0]
                break
        if domain_type is not None:
            for key, value in config.OF_case_data_dict.items():
                if domain_type == key.split('/')[0]:
                    file_alternative[key.split("/")[-1]] = set(value['configuration_files'].keys())
                    file_turbulence_model[key.split("/")[-1]] = value["turbulence_model"]

    if file_turbulence_model != {}:
        # Process files caused by turbulence model differences
        turbulence_path = f'{config.Database_OFv24_PATH}/final_OF_turbulence_required_files.json'
        with open(turbulence_path, 'r', encoding='utf-8') as f:
            turbulence_files_structure = json.load(f)
        # print(file_turbulence_model)
        for case_name, turbulence_model_name in file_turbulence_model.items():
            if turbulence_model_name != None:
                excessive_file = set(turbulence_files_structure[turbulence_model_name]) - set(turbulence_files_structure[turbulence_model])
                missing_file = set(turbulence_files_structure[turbulence_model]) - set(turbulence_files_structure[turbulence_model_name])
                file_alternative[case_name] = (file_alternative[case_name] - excessive_file).union(missing_file)

    for case_name, file_list in file_alternative.items():
        # Filter out unnecessary system files, keep only required files
        filtered_files = {file for file in file_list 
                        if not (file.startswith("system/") and file not in system_necessary)}
        
        filtered_files.add("system/blockMeshDict")
        # Update the file list
        file_alternative[case_name] = filtered_files

    gian_file_structure = f"""You are an OpenFOAM expert. The simulation requirement is: {config.simulate_requirement}

    Below are the optional reference cases and their file lists:
    {file_alternative}

    1. Based on the simulation requirement, identify the file list that best matches the requirement and return the corresponding reference case name.
    2. If no suitable file list exists, provide the closest reference case name.
    3. If all file lists are completely unsuitable, return "none".
    4. Return only the case name, with no explanations, code blocks, or additional content.
    """

    qa = QA_NoContext_deepseek_V3()

    if file_alternative != {}:
        print("Finding suitable file structure...")
        # appropriate_files = None
        get_name = False
        max_retries = 3  # Set maximum retry count
        retry_count = 0
        while retry_count < max_retries and not get_name:
            try:
                response = qa.ask(gian_file_structure).strip()
                # print(response)
                # print(file_alternative)
                if response.lower() != "none":
                    file_structure = file_alternative[response]
                    get_name = True
                else:
                    get_name = True
            except KeyError:
                print(f"Reference case name error: {response}")
            retry_count += 1

        if not get_name:
            response.lower() == "none"
    else:
        print("No reference cases available")
        response = "none"

    if response.lower() == "none":
        print("No suitable reference case found, using default file list")
        
        if config.case_turbulence_model in ["SpalartAllmarasDDES", "SpalartAllmarasIDDES"]:
            config.case_turbulence_type = "LES"
        elif config.case_turbulence_model in ["SpalartAllmaras","kOmegaSST","LaunderSharmaKE","realizableKE","kOmegaSSTLM","kEpsilon","RNGkEpsilon"]:
            config.case_turbulence_type = "RAS"
        else:
            config.case_turbulence_type = "laminar"

        with open(f"{config.Database_OFv24_PATH}/final_OF_solver_required_files.json", 'r', encoding='utf-8') as file:
            file_structure = set(json.load(file)[solver])
        if turbulence_model != None:
            with open(f"{config.Database_OFv24_PATH}/final_OF_turbulence_required_files.json", 'r', encoding='utf-8') as file:
                file_structure = file_structure.union(set(json.load(file)[turbulence_model]))

    file_structure.discard("system/blockMeshDict")
    # config.global_files = list(file_structure)
    # file_structure = list(file_structure)
    if file_turbulence_model == {}:
        for k in file_alternative.keys():
            file_turbulence_model[k] = turbulence_model

    file_name = response
    print(file_name) # Reference name
    print(file_structure)   
    print(file_turbulence_model)    # May still differ from the required turbulence model
    # Can consider special models further
    return file_structure, file_name, file_turbulence_model

# Generate initial files
def generate_initial_files(case_description=None, output_case_path=None, grid_path=None, grid_type=None):
    if case_description is None:
        case_description = config.case_info.case_description
    if output_case_path is None:
        output_case_path = config.path_cfg.output_case_path
    if grid_path is None:
        grid_path = config.path_cfg.grid_path
    if grid_type is None:
        grid_type = config.grid_type

    print("Generating controlDict, mesh conversion, extracting mesh boundary information...")
    write_controlDict = False
    max_retries = 3  # Set maximum retry count
    retry_count = 0
    while not write_controlDict and retry_count < max_retries:
        try:
            # Generate controlDict
            file_content = f"""FoamFile
{{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      controlDict;
}}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

application     {config.case_info.case_solver};

startFrom       startTime;

startTime       0;

stopAt          endTime;

endTime         0.1;

deltaT          0.01;

writeControl    timeStep;

writeInterval   20;

purgeWrite      20;

writeFormat     ascii;

writePrecision  6;

writeCompression off;

timeFormat      general;

timePrecision   6;

runTimeModifiable true;"""

            if not os.path.exists(os.path.join(output_case_path, "system/controlDict")):
                os.makedirs(os.path.join(output_case_path, "system"), exist_ok=True)
            with open(os.path.join(output_case_path, "system/controlDict"), "w") as f:
                f.write(file_content)

            # setup_cfl_control(case_path=output_case_path, max_co=0.6)
            # Mesh conversion
            convert_mesh(output_case_path=output_case_path, grid_path=grid_path, grid_type=grid_type)
            # Extract mesh boundary names and conditions (after mesh conversion, grid_type becomes polyMesh)
            config.grid_info.grid_boundary_conditions = extract_boundary_names(grid_file_path=os.path.join(output_case_path, "constant/polyMesh"), grid_type='polyMesh')
            print("Mesh boundaries:", config.grid_info.grid_boundary_conditions)
            write_controlDict = True
        except Exception as e:
            print(f"Unexpected error: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
            retry_count += 1
            if retry_count < max_retries:
                print(f"Retrying... ({retry_count}/{max_retries})")
            else:
                print("Maximum retry attempts reached, terminating operation")
                raise

    # Check if initialization completed successfully
    if not write_controlDict:
        error_msg = "Initialization failed: unable to complete controlDict generation, mesh conversion, or boundary condition extraction"
        print(f"Error: {error_msg}")
        raise RuntimeError(error_msg)

    # Process PDF or txt, and use RAG to answer questions
    extractor = pdf_chunk_ask_question.CFDCaseExtractor()
    extractor.process_pdf(config.pdf_path)

    # Get physical fields
    initial_files = []  # Field files
    for i in config.case_info.file_structure:
        if "0/" in i:
            initial_files.append(i[2:])

    # Get boundary condition settings from PDF or txt
    bc_template = {}
    for i in initial_files:
        bc_template[i] = {}
        for j in config.grid_info.grid_boundary_conditions.keys():
            bc_template[i][j] = ""
    print("Boundary condition template:", bc_template)

    query_embedded = f"Could you please clarify what the boundary conditions are for the {config.case_name} example mentioned in the text?"
    get_bc_prompt = f"""{query_embedded}Please return the boundary-condition settings for each physical field on every boundary of this case based on the following information:

Available boundary-condition types:
<boundary_conditions>
{config.string_of_boundary_type_keywords}
</boundary_conditions>

Boundary names and their geometric types from the mesh:
<mesh_boundary_conditions>
{config.grid_info.grid_boundary_conditions}
</mesh_boundary_conditions>

<output_requirements>
1. Complete the boundary-condition details for every physical field and return the full content in JSON format, without any additional text:
{bc_template}
2. Verify and correct any spelling errors in boundary-condition types against the provided "available boundary-condition types".
3. If a boundary name contains a slash (e.g., a/b), split it into separate boundaries ("a" and "b") and list them individually.
4. Boundary names in the physical-field specifications must match those in the mesh boundary conditions, and the chosen boundary-condition types must not conflict with the geometric boundary types.
5. If the CFD case excerpt does not specify a boundary-condition setting, leave it as an empty string.
</output_requirements>"""

    bc_response = extractor.query_case_setup(query_embedded, get_bc_prompt, context = True)
    # print("Boundary condition settings:", bc_response)
    bc_info = extract_content_from_response(bc_response, 'json')
    # print("Boundary condition settings:", bc_info)

    with open(os.path.join(config.path_cfg.database_dir,"OF_bc_entry.json"), 'r') as f:
        OF_bc_entry = json.load(f)

    ic_bc_template = {}
    for i in initial_files:
        ic_bc_template[i] = {"internalField":"", "boundaryField":{}}
        for j in config.grid_info.grid_boundary_conditions.keys():
            ic_bc_template[i]["boundaryField"][j] = {"type":bc_info[i][j]}
            if bc_info[i][j] in OF_bc_entry.keys():
                for k in OF_bc_entry[bc_info[i][j]]:
                    ic_bc_template[i]["boundaryField"][j][k] = ""
            else:
                continue   # noSlip, zeroGridient, empty

    # print("Initial condition and boundary condition template:", ic_bc_template)
    query_embedded = f"What are the initial conditions for {initial_files} in the {config.case_name} case described in the document?"
    get_ic_bc_prompt = f"""{query_embedded}Please return the initial and boundary-condition settings for all physical fields in this case according to the following requirements:

<output_requirements>
1. Complete the initial-condition details for every physical field and return the full content in JSON format only:
{ic_bc_template}
2. Ensure all initial values conform to OpenFOAM syntax conventions.
3. When filling 'internalField' or the 'value' entry on boundaries, use uniform initial values unless otherwise specified; for vector quantities, list each component—for example, (0 0 0).
4. If the CFD case excerpt does not specify any boundary or initial condition, leave the corresponding entry as an empty string.
</output_requirements>"""
    
    ic_bc_response = extractor.query_case_setup(query_embedded, get_ic_bc_prompt, context = True)
    # print("Boundary condition settings:", bc_response)
    ic_bc_info = extract_content_from_response(ic_bc_response, 'json')
    print("Initial condition and boundary condition settings:", ic_bc_info)
    config.grid_info.field_ic_bc_from_input = ic_bc_info

    # other_setting_info = ""
    # print("Discretization schemes, solver settings, material properties, etc.:")

    # Preparation before generating OpenFOAM case files
    class FileContent(BaseModel):
        files_content: Dict[str, str] = Field(description="A mapping from file name to its file content")
    parser = PydanticOutputParser(pydantic_object=FileContent)

    OF_header = '''/*--------------------------------*- C++ -*----------------------------------*\\\\\\n| =========                 |                                                 |\\n| \\\\\\\\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           |\\n|  \\\\\\\\    /   O peration     | Version:  v2406                                 |\\n|   \\\\\\\\  /    A nd           | Website:  www.openfoam.com                      |\\n|    \\\\\\\\/     M anipulation  |                                                 |\\n\\\\*---------------------------------------------------------------------------*/'''
    Foamfile_string = '''FoamFile
{
    version     2.0;
    format      ascii;
    class       <the_file_class_type>;
    object      <the_file_object_type>;
}
'''

    zero_files = []
    constant_files = []
    system_files = []
    multiple_dimensions = {}
    for i in config.case_info.file_structure:
        if i.startswith("0/"):
            zero_files.append(i)
            if i in ["0/p", "0/alphat_", "0/p_gh","0/B_", "0/pa"]:
                with open(os.path.join(config.path_cfg.database_dir, 'OF_case_dimensions.json'), 'r', encoding='utf-8') as f:
                    dimensions_dict = json.load(f)
                multiple_dimensions[i] = [dimensions_dict[i], dimensions_dict[i+"_"]]
        elif i.startswith("constant/"):
            constant_files.append(i)
        elif i.startswith("system/"):
            if i == "system/controlDict":    # Skip controlDict
                continue
            system_files.append(i)

    # Prepare reference file content in advance
    reference_files = {}
    if config.case_info.reference_file_name != "":
        with open(os.path.join(config.path_cfg.database_dir, 'processed_merged_OF_cases.json'), 'r', encoding='utf-8') as f:
            reference_files = json.load(f)
            for file_name in reference_files:
                if config.case_info.reference_file_name == file_name.split("/")[-1]:
                    reference_files = reference_files[file_name]["configuration_files"]
                    config.case_info.reference_file_name = file_name
                    break
            else:
                reference_files = {}

    print(f"Reference file when generating initial files: {config.case_info.reference_file_name}")
    reference_files_zero = {}
    reference_files_constant = {}
    reference_files_system = {}
    controlDict_ref = None
    for file_name in reference_files:
        if file_name.startswith("0/") and file_name in zero_files:
            reference_files_zero[file_name] = reference_files[file_name]
        elif file_name.startswith("constant/") and file_name in constant_files:
            reference_files_constant[file_name] = reference_files[file_name]
        elif file_name == "system/controlDict":
            # print("system/controlDict")
            # print(reference_files[file_name])
            controlDict_ref = reference_files[file_name]
        elif file_name.startswith("system/") and file_name in system_files:
            reference_files_system[file_name] = reference_files[file_name]

    setup_cfl_control(case_path=output_case_path, max_co=0.6, controlDict_ref=controlDict_ref)

    qa = QA_NoContext_deepseek_R1()
    generate_files_prompt_0 = f"""I would like to simulate the following case with OpenFOAM-v2406:

<case_requirements>
- Solver: {config.case_info.case_solver}
- Turbulence model: {config.case_info.turbulence_model}
- Case description: {case_description}
</case_requirements>

<initial_and_boundary_conditions>
{ic_bc_info}
</initial_and_boundary_conditions>

Please generate the file contents for:
<file_to_be_generated>
{zero_files}
</file_to_be_generated>

Fields that may adopt multiple dimensions:
<multiple_dimensions>
{multiple_dimensions}
</multiple_dimensions>

Reference files from the OpenFOAM tutorials:
<reference_files>
{reference_files_zero}
</reference_files>

Please follow the requirements below for your output:
<output_requirements>
1. Return the results strictly in the following JSON format, without any additional content.:
{parser.get_format_instructions()}
2. Ensure that every boundary name is explicitly configured in every physical-field file to guarantee correctness and completeness.
3. All physical-field files must comply with the provided initial and boundary conditions; where these are unspecified, make reasonable inferences based on the case description.
4. Boundary-condition settings in the physical-field files must not conflict with those in the mesh, nor with those in other physical-field files.
5. For fields that admit alternative dimensions, select the appropriate set according to the simulation requirements (e.g., use [0 2 -2 0 0 0 0] for pressure in incompressible flow and [1 -1 -2 0 0 0 0] for compressible flow).
6. Do not include the OpenFOAM file-header line (e.g., {OF_header}) for each file, but retain the FoamFile block (e.g., {Foamfile_string}).
7. The output must be complete and self-contained; do not rely on #include directives to pull in external content.
</output_requirements>"""
    case_file = parser.parse(extractor.query_case_setup(generate_files_prompt_0, context = True)).files_content
    # case_file = parser.parse(qa.ask(generate_files_prompt_0)).files_content

    reference_files_constant.update(reference_files_system)
    generate_files_prompt_1 = f"""I would like to simulate the following case with OpenFOAM-v2406:

<case_requirements>
- Solver: {config.case_info.case_solver}
- Turbulence model: {config.case_info.turbulence_model}
- Case description: {case_description}
- List of physical fields: {initial_files}
</case_requirements>

Please generate the contents for the following files:
<file_to_be_generated>
{constant_files + system_files}
</file_to_be_generated>

Reference files from the OpenFOAM tutorials:
<reference_files>
{reference_files_constant}
</reference_files>

Please follow the requirements below for your output:
<output_requirements>
1. Return the results strictly in the following JSON format, without any additional content.:
{parser.get_format_instructions()}
2. While generating each file, verify that the settings are reasonable and that no content is missing or superfluous.
3. Do not include the OpenFOAM file-header line (e.g., {OF_header}) for each file, but retain the FoamFile block (e.g., {Foamfile_string}).
4. The output must be complete and self-contained; do not rely on #include directives to pull in external content.
</output_requirements>"""

    case_file.update(parser.parse(extractor.query_case_setup(generate_files_prompt_1, context = True)).files_content)
    # case_file.update(parser.parse(qa.ask(generate_files_prompt_1)).files_content)
    # print(case_file)

    # print(json.dumps(case_file, indent=4, ensure_ascii=False))
    return case_file

def check_file_format(files_content=None):
    """Reference OpenFOAM example files to check if there are formatting issues
    Args:
        files_content (dict): File content, key is file name, value is file content
    Returns:
        files_content (dict): Modified file content
    """
    if files_content == None:
        files_content = config.global_files

    check_file_prompt = """Please cross-check the following {file_name} file against the same file in other OpenFOAM-tutorials cases for any formatting issues. Focus on file structure, nesting logic, completeness and correctness of keywords, and whether any statements are missing.

<file_content>
{file_content}
</file_content>

<openfoam_reference_files>
{reference_files}
</openfoam_reference_files>

<output_requirement>
1. If formatting issues are found, return the corrected, complete file content enclosed in ``` and ``` only—do not include explanations or reasoning.
2. If the format is correct, simply reply NO.
</output_requirement>"""
    with open(os.path.join(config.path_cfg.database_dir, 'OF_case_dimensions.json'), 'r', encoding='utf-8') as f:
        dimensions_dict = json.load(f)

    for file_name, file_content in files_content.items():
        # Check file format
        if file_name.startswith("constant/") or file_name.startswith("system/"):
            print(f"Checking {file_name} file format...")
            qa = QA_NoContext_deepseek_V3()
            reference_files = find_reference_files(file_name)

            response = qa.ask(check_file_prompt.format(file_name=file_name, file_content=file_content, reference_files=reference_files))

            if "NO" in response:
                continue
            else:
                files_content[file_name] = extract_content_from_response(response, "str")

        # Check dimensions
        if file_name in dimensions_dict.keys():
            if "dimensions" in file_content:
                dimension_pattern = r'dimensions.*?\n'
                match = re.search(dimension_pattern, file_content)
                if match:
                    if file_name+"_" in dimensions_dict.keys(): # Underscore suffix for incompressible
                        if file_name in ["0/p", "0/p_rgh", "0/alphat"]: 
                            if config.case_info.case_solver in config.compressible_solvers or "compressible" in file_content:
                                files_content[file_name] = re.sub(dimension_pattern, f'dimensions      {dimensions_dict[file_name]};\n', file_content)
                            elif config.case_info.case_solver in config.incompressible_solvers:
                                files_content[file_name] = re.sub(dimension_pattern, f'dimensions      {dimensions_dict[file_name+"_"]};\n', file_content)
                            else:
                                continue    
                        else:
                            continue    # Skip decision, let LLM set according to initially generated content
                    else:
                        files_content[file_name] = re.sub(dimension_pattern, f'dimensions      {dimensions_dict[file_name]};\n', file_content)
            else:
                if file_name+"_" in dimensions_dict.keys(): 
                     if file_name in ["0/p", "0/p_rgh", "0/alphat"]: 
                         if config.case_info.case_solver in config.compressible_solvers or "compressible" in file_content:
                             files_content[file_name] = f"\ndimensions      {dimensions_dict[file_name]};\n"
                         elif config.case_info.case_solver in config.incompressible_solvers:
                             files_content[file_name] = f"\ndimensions      {dimensions_dict[file_name+'_']};\n"
                     else:
                         continue    # File will error, let LLM solve it itself
                else:
                    files_content[file_name] += f"\ndimensions      {dimensions_dict[file_name]};\n"
    
    return files_content

