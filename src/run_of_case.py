import subprocess
from PyFoam.RunDictionary.ParsedParameterFile import ParsedParameterFile

import config

"""
May be removed later
"""

def convert_mesh(case_path, grid_path):
    """Convert Fluent mesh to OpenFOAM format"""
    try:
        command = [
            "fluentMeshToFoam",
            "-case",
            case_path,
            grid_path
        ]
        subprocess.run(command, check=True)
        print("Mesh conversion completed successfully")
        config.mesh_convert_success = True
        return True
    except subprocess.CalledProcessError as e:
        print(f"Mesh conversion failed: {e}")
        return False
    except FileNotFoundError:
        print("fluentMeshToFoam command not found, please ensure OpenFOAM environment is properly loaded")
        return False

def setup_cfl_control(case_path, max_co=0.6):
    """Set CFL control parameters"""
    try:
        # Modify controlDict file
        control_dict_path = f'{case_path}/system/controlDict'
        control_dict = ParsedParameterFile(control_dict_path)

        demo_compressible_solver = ["rhoCentralFoam", "sonicFoam"]

        solver = control_dict["application"]
        if solver in config.steady_solvers:
            control_dict["adjustTimeStep"] = "yes"
            control_dict["maxCo"] = max_co
            control_dict["startTime"] = 0
            control_dict["endTime"] = 10
            control_dict["stopAt"] = "endTime"
            control_dict["writeInterval"] = 5
            control_dict["deltaT"] = 1
        else:
            control_dict["adjustTimeStep"] = "yes"
            control_dict["maxCo"] = max_co
            control_dict["startTime"] = 0
            dt = 1e-8
            if solver in demo_compressible_solver:
                dt = 1e-8
            else:
                dt = 1e-5
            control_dict["deltaT"] = dt
            control_dict["endTime"] = dt*10
            control_dict["stopAt"] = "endTime"
            control_dict["writeInterval"] = 2
            if solver in demo_compressible_solver:
                control_dict["deltaT"] = 1e-8
            else:
                control_dict["deltaT"] = 1e-5
        
        # Save modifications
        control_dict.writeFile()
        config.set_controlDict_time = True
        print("Successfully configured CFL control parameters")
        return True
    except Exception as e:
        print(f"Failed to modify controlDict: {e}")
        return False
    
def setup_cfl_control_2(case_path, max_co=0.6):
    """Set CFL control parameters"""
    try:
        # Modify controlDict file
        control_dict_path = f'{case_path}/system/controlDict'
        control_dict = ParsedParameterFile(control_dict_path)
        control_dict["adjustTimeStep"] = "yes"
        control_dict["maxCo"] = max_co
        control_dict["startTime"] = 0
        control_dict["endTime"] = 2
        control_dict["stopAt"] = "endTime"
        control_dict["writeInterval"] = 1
        control_dict["deltaT"] = 1
        
        # Save modifications
        control_dict.writeFile()
        print("Successfully configured CFL control parameters")
        return True
    except Exception as e:
        print(f"Failed to modify controlDict: {e}")
        return False

def case_run(case_path):
    solver = ""
    try:
        control_dict_path = f'{case_path}/system/controlDict'
        # control_dict = ParsedParameterFile(control_dict_path)
        # solver = control_dict["application"]
        # Open file and read content
        with open(control_dict_path, 'r') as file:
            content = file.read()

        # Find content after "application"
        start_index = content.find('application') + len('application')
        end_index = content.find(';', start_index)

        # Extract string and remove whitespace
        solver = content[start_index:end_index].strip()
    except Exception as e:
        print(f"Fail acquiring the solver: {e}")
        return False

    running_log = f'{case_path}/case_run.log'

    command = f'{solver} -case {case_path} > {running_log}'
    # command = f'ls'
    output = subprocess.run(
        command,
        shell=True,
        executable="/usr/bin/bash",
        text=True,
        capture_output=True  # get stdout and stderr
        )
    
    run_case_error = output.stderr
    run_case_output = output.stdout
    
    if output.returncode != 0: # Check if command execution resulted in an error
        print("Program error! Error message:", output.stderr)
        return output.stderr
    else:
        print("Program ran successfully, output:", output.stdout)
        config.flag_case_success_run = True
        return "case run success."


    # output_dir = Path(opts.output_dir.get_path())
