import os
import json
import torch
import pdfplumber

import config, file_writer, run_of_case, file_corrector,file_preparation

import Reflextion

torch.classes.__path__ = [os.path.join(torch.__path__[0], torch.classes.__file__)]

def process_pdf_pdfplumber(file_path):
    """Extract PDF text and tables using pdfplumber"""
    text = ""
    tables = []

    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            # Extract text
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"

            # Extract tables
            page_tables = page.extract_tables()
            if page_tables:
                for table in page_tables:
                    tables.append(table)

    return {
        "text": text,
        "tables": tables
    }

def load_OF_data_json():
    try:
        with open(config.OF_data_path, 'r', encoding='utf-8') as file:
            full_data = json.load(file)
            config.OF_case_data_dict = {}
            for case_path, case_info in full_data.items():
                if config.case_info.case_solver in case_path:
                    config.OF_case_data_dict[case_path] = case_info
            print("Successfully read the OF_tut_case_json file!")
    except json.JSONDecodeError:
        print("Input JSON format error, please check data integrity")
        exit()

def main(case_name_idx):

    # Load PDF or txt file
    if config.pdf_path.endswith('.pdf'):
        config.paper_content, config.paper_table = process_pdf_pdfplumber(config.pdf_path)
    else:
        with open(config.pdf_path, 'r', encoding='utf-8') as file:
            config.paper_content = file.read()
            config.paper_table = []

    # prepare config
    config.OUTPUT_PATH = os.path.join(config.path_cfg.output_path, case_name_idx)
    config.path_cfg.output_case_path = os.path.join(config.path_cfg.output_path, case_name_idx)

    # Create folder for storing cases
    config.ensure_directory_exists(config.path_cfg.output_case_path)
    config.case_log_write = True

    # list the files required by the solver and turbulence model
    config.global_files, config.case_info.reference_file_name, _ = file_preparation.case_required_files(config.case_info.case_solver, config.case_info.turbulence_model)
    config.global_files = list(config.global_files)
    config.case_info.file_structure = list(config.global_files)
    print("File structure:", config.case_info.file_structure)

    # Generate initial files
    write_initial_files = False
    while not write_initial_files:
        try:
            config.global_files = file_preparation.generate_initial_files()
            write_initial_files = True
        except:
            print("Regenerating initial files")

    # Simple check of file format and ensure correct dimensions
    print("Performing simple checks...")
    config.global_files = file_preparation.check_file_format(config.global_files)

    # write the case files
    for key, value in config.global_files.items():
        output_file = f"{config.path_cfg.output_case_path}/{key}"

        try:
            file_preparation.write_field_to_file(value,output_file)
            print(f"write the file {key}")

        except Exception as e:
            print(f"Errors occur during write_field_to_file: {e}")
            continue

    with open(f"{config.path_cfg.output_case_path}/error_history.txt", "w") as f:
        f.write("****************error_history****************\n")

    # run the OpenFOAM case and ICOT debug
    for test_time in range(0, config.max_running_test_round):
        try:
            print(f"****************start running the case {case_name_idx} , test_round = {test_time}****************")

            case_run_info = run_of_case.case_run(config.path_cfg.output_case_path)    # Run OpenFOAM case using subprocess
            
            if case_run_info != "case run success.":
                running_error = case_run_info
                # Error history record
                with open(f"{config.path_cfg.output_case_path}/error_history.txt", "a") as f:
                    f.write(f"=====Test round {test_time}=====\nRunning error:\n{running_error}\n")

                need_reflextion = False
                config.error_history.append(running_error)
                if len(config.error_history) > 4:
                    config.error_history = config.error_history[-4:]  # Keep only the latest 4 entries
                    config.correct_trajectory = config.correct_trajectory[-4:]

                if len(config.error_history) > 1:
                    last_error = config.error_history[-1]  # Last error
                    count = 1  # Same error count, the last error itself counts as 1
                    for i in range(len(config.error_history) - 2, -1, -1):
                        if config.error_history[i] == last_error:
                            count += 1
                        else:
                            break  # Stop when encountering different errors
                    if count >= 4:  # Same error occurred 4 times (reflected twice but still failed), rewrite file
                        file_for_revision, early_revision_advice = file_corrector.analyze_running_error_with_all_case_file_content(running_error)
                        reference_files = file_corrector.find_reference_files_by_solver(file_for_revision)  # Find reference files based on file_for_revision
                        print("Rewriting file")
                        file_corrector.rewrite_file(file_for_revision,reference_files)
                        config.error_history = []  # Reset error history
                        with open(f"{config.path_cfg.output_case_path}/error_history.txt", "a") as f:
                            f.write("Error correction plan:\nRewrite file\n")
                    elif count > 1:
                        # Same error occurred consecutively, start reflection
                        reflection_result = Reflextion.reflextion(running_error, config.correct_trajectory[-1*count:])
                        relevant_reflections = Reflextion.constructe_reflection_context(running_error, Reflextion.reflection_history)
                        need_reflextion = True

                if need_reflextion == False:
                    relevant_reflections = ""

                answer_add_new_file = file_corrector.identify_error_to_add_new_file(running_error, relevant_reflections)
                answer_add_new_file_strip = answer_add_new_file.strip()
                
                if answer_add_new_file_strip.lower() != 'no':
                    print("Adding missing files")
                    file_for_adding = answer_add_new_file_strip
                    config.correct_trajectory.append({file_for_adding:[file_corrector.add_new_file(file_for_adding)]})

                    with open(f"{config.path_cfg.output_case_path}/error_history.txt", "a") as f:
                        f.write(f"Error correction plan:\nAdd file {file_for_adding}\n")
                else:
                    error_files = file_corrector.analyse_error(running_error, config.case_info.file_structure, relevant_reflections)
                    config.correct_trajectory.append(file_corrector.correct_error(running_error, error_files, config.case_info.file_structure, relevant_reflections))

                    try:
                        with open(f"{config.path_cfg.output_case_path}/error_history.txt", "a") as f:
                            f.write(f"Error correction plan:\nModify files {config.correct_trajectory[-1].keys()}\n")
                    except:
                        print("Error correction plan:\nModify files...")

                if not config.set_controlDict_time:
                    run_of_case.setup_cfl_control(config.path_cfg.output_case_path)

                if not config.mesh_convert_success:
                    file_preparation.convert_mesh(config.path_cfg.output_case_path, config.case_grid)

            else:
                with open(f"{config.path_cfg.output_case_path}/cycle_index.txt", "w") as f:
                    f.write(f"Case {case_name_idx} run successfully at test_round {test_time}+1.\n")

                break
                
        except Exception as e:
            try:
                import traceback
                traceback.print_exc()
                # -------------- Exception handling --------------

                # Catch and handle all exceptions
                running_error = str(e)
                with open(f"{config.path_cfg.output_case_path}/error_history.txt", "a") as f:
                    f.write(f"Runtime error occurred: {running_error}\n")
                print("running_error: ", running_error)

                # Determine if new files need to be added
                answer_add_new_file = file_corrector.identify_error_to_add_new_file(running_error)

                answer_add_new_file_strip = answer_add_new_file.strip()

                if answer_add_new_file_strip.lower() == 'no':# File modification branch

                    file_for_revision, early_revision_advice = file_corrector.analyze_running_error_with_all_case_file_content(running_error)
                    reference_files = file_corrector.find_reference_files_by_solver(file_for_revision)
                    # Check if error occurred three times, if so, rewrite the file.
                    if file_corrector.analyze_error_repetition(config.error_history):
                        file_corrector.rewrite_file(file_for_revision,reference_files)
                    else:
                        advices_for_revision = file_corrector.analyze_running_error_with_reference_files(running_error, file_for_revision,early_revision_advice,reference_files)
                        file_corrector.single_file_corrector2(file_for_revision, advices_for_revision, reference_files)
                else:# File addition branch
                    file_for_adding = answer_add_new_file_strip
                    file_corrector.add_new_file(file_for_adding)

                if not config.set_controlDict_time:
                    run_of_case.setup_cfl_control(config.path_cfg.output_case_path)

                if not config.mesh_convert_success:
                    run_of_case.convert_mesh(config.path_cfg.output_case_path, config.case_grid)
            except Exception as e:
                print(f"Errors occur during exception handling: {e}")

            continue  # Explicitly continue to next loop

def run_case():
    load_OF_data_json()

    # Run 10 times
    run_times = config.run_cfg.run_time

    for i in range(run_times):
        print(f"Simulation {i+1}")
        case_name = f"{config.case_info.case_name}_{i}"

        main(case_name)
