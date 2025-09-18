import os
import sys
import json
import pathlib
import argparse
from datetime import datetime

import torch
import PyPDF2
import tiktoken
import streamlit as st
from openai import OpenAI

import config, preprocess_OF_tutorial, main_run_chatcfd, qa_modules, file_preparation

torch.classes.__path__ = [os.path.join(torch.__path__[0], torch.classes.__file__)] 


# chatbot prompt
SYSTEM_PROMPT = """You are an intelligent assistant capable of:
1. Maintaining politeness and professionalism
2. Remembering the context of the conversation
3. Processing and analyzing content from documents uploaded by users
4. Answering user questions while keeping the conversation coherent

Please always respond in a clear, accurate, and helpful manner
"""

EXTRACT_CASES_FROM_TEXT_PROMPT = """The attached PDF contain several CFD cases, and I would like to run one or several of the case by my self later. Please read the paper and list all distinct CFD cases with characteristic description. Give each case a tag as Case_X (such as Case_1, Case_2).

- Please count each unique combination of parameters that results in a separate simulation run as one CFD case. These parameters include but not limited to the geometry, boundary Conditions, flow Parameters (Re/Mach/AoA/velocity), physical Model, or Solver.
- If there are multiple runs of the same parameters for statistical analysis or convergence studies, count these as one case, unless the paper specifies them as distinct due to different goals or conditions.
- If any case is simulated using OpenFOAM, identify the solver or find a proper solver to run the case. Show the solver name when describing the case.

The paper is as follows: 
{text_content}.
"""

JSON_RESPONSE_SAMPLE = '''
{
    "Case_1":{
        "case_name":"<some_case_name>",
        "solver":"<solver_name>",
        "turbulence_model":"<model_name>",
        "other_physical_model":"<model_name>",
        "case_specific_description":"<a sentence that describes the case setup with detailed parameters that differenciate this case from the other cases in the paper>"
    }
}
'''

ASK_TO_CHOOSE_CASE_AND_SOLVER = """Please choose the case you want to simulate and the OpenFOAM solver you want to use. 
Your answer shall be like one of the followings:
- I want to simulate Case_1 using rhoCentralFoam and the SpalartAllmaras model.
- I want to simulate the Case with AOA = 10 degree and kOmegaSST model.
        
You must choose only one case.
"""

CONFIRM_SIMULATION_REQUIREMENT = """Next, the CFD simulation will be conducted according to the following settings:
{simulate_requirement}
Do you confirm this simulation setting? If you do, please reply with "yes".
"""

GUIDE_CASE_CHOOSE_PROMPT = """Understand the user's answer and describe the case details of the user's requirement.

    The user's answer is: {user_answer}

    Please generate JSON content according to these requirements:

    1. Strictly follow this example format containing ONLY JSON content: {json_response_sample}

    2. Absolutely AVOID any non-JSON elements including but not limited to:
    - Markdown code block markers (```json or ```)
    - Extra comments or explanations
    - Unnecessary empty lines or indentation
    - Any text outside JSON structure

    3. Critical syntax requirements:
    - Maintain strict JSON syntax compliance
    - Enclose all keys in double quotes
    - Use double quotes for string values
    - Ensure no trailing comma after last property

    4. Case_name must adhere to the following format:
        [a-zA-Z0-9_]+ - only containing lowercase letters, uppercase letters, numbers, or underscores. Special characters (e.g. -, @, #, spaces) are not permitted.

    5. The solver must be one of the followings: {string_of_solver_keywords}. 
    The turbulence_model must be one of the followings: {string_of_turbulence_model}.
    If a case employs the laminar flow assumption, then the turbulence_model is set to 'null'.
"""


# OpenFOAM case setup keywords, used to extract keywords from case descriptions to match the most relevant cases in the case library. OF-v2406

class ChatBot:
    def __init__(self):
        self.client = OpenAI(
            api_key=os.environ.get("DEEPSEEK_R1_KEY"),
            base_url=os.environ.get("DEEPSEEK_R1_BASE_URL")
        )
        self.system_prompt = SYSTEM_PROMPT
        self.temperature = 0.9

        self.token_counter = {
            "total": 0,
            "qa_history": []
        }

    def process_pdf(self, pdf_file):
        try:
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text()
            return text
        except Exception as e:
            return f"PDF processing error: {str(e)}"

    def get_response(self, messages):

        try:
            response = self.client.chat.completions.create(
                model=os.environ.get("DEEPSEEK_R1_MODEL_NAME"),
                messages=[{"role": "system", "content": self.system_prompt}] + messages,
                temperature=self.temperature
            )
            # Record token usage
            usage = response.usage
            self.token_counter["total"] += usage.total_tokens
            # qa_record = {
            #     "prompt": messages,
            #     "prompt_tokens": usage.prompt_tokens,
            #     "completion_tokens": usage.completion_tokens,
            #     "total_tokens": usage.total_tokens,
            #     "timestamp": datetime.now().isoformat()
            # }
            return response.choices[0].message.content
        except Exception as e:
            return f"Chat error: {str(e)}"

    def count_tokens(self, text: str, model: str = "gpt-4o") -> int:
        """Count token numbers using tiktoken"""
        try:
            encoding = tiktoken.encoding_for_model(model)
            return len(encoding.encode(text))
        except KeyError:
            encoding = tiktoken.get_encoding("cl100k_base")
            return len(encoding.encode(text))

def initialize_session_state():
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "chatbot" not in st.session_state:
        st.session_state.chatbot = ChatBot()
    if "file_content" not in st.session_state:
        st.session_state.file_content = None
    if "file_processed" not in st.session_state:
        st.session_state.file_processed = False
    if "ask_case_solver" not in st.session_state:
        st.session_state.ask_case_solver = False
    # if "user_answered" not in st.session_state:
    #     st.session_state.user_answered = False
    if "user_answer_finished" not in st.session_state:
        st.session_state.user_answer_finished = False
    if "uploaded_grid" not in st.session_state:
        st.session_state.uploaded_grid = False
    if "show_start" not in st.session_state:
        st.session_state.show_start = False

def main():
    # streamlit functions

    st.title("ChatCFD: chat to run CFD cases.")

    st.divider()
    
    initialize_session_state()

    with st.sidebar:

        # Export chat history functionality
        st.header("Export chat history")
        export_format = "JSON"
        
        if st.button("Export chat"):
            if not st.session_state.messages:
                st.warning("Empty chat history")
            else:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"chatlog_{timestamp}"

                chat_data = {
                    "metadata": {
                        "export_time": datetime.now().isoformat(),
                        "total_messages": len(st.session_state.messages),
                        "total_tokens": st.session_state.chatbot.token_counter["total"]
                    },
                    "messages": st.session_state.messages
                }
                
                st.sidebar.download_button(
                    label="Download JSON file",
                    data=json.dumps(chat_data, indent=2, ensure_ascii=False),
                    file_name=f"{filename}.json",
                    mime="application/json"
                )

    # Sidebar: File Upload
    with st.sidebar:
        st.header("Upload the document")
        uploaded_file = st.file_uploader(
            "Please upload PDF",
            type=['pdf']
        )
        
        if uploaded_file:
            if not st.session_state.file_processed:
                if uploaded_file.type == "application/pdf":

                    save_dir = pathlib.Path(config.TEMP_PATH)
                    
                    try:
                        # Build save path
                        file_path = save_dir / uploaded_file.name.replace(" ", "_")
                        
                        # Save uploaded PDF file
                        with open(file_path, "wb") as f:
                            f.write(uploaded_file.getbuffer())

                        config.pdf_path =  f"{config.TEMP_PATH}/{uploaded_file.name}"
                        config.path_cfg.case_description_path = config.pdf_path
                        config.case_info.file_name = os.path.basename(config.path_cfg.case_description_path).rstrip("txt").rstrip("pdf").rstrip(".")
                        config.path_cfg.output_path = os.path.join(config.path_cfg.output_dir, config.case_info.file_name)

                    except Exception as e:
                        st.error(f"Failed at processed the pdf file: {str(e)}") 

                    text_content = st.session_state.chatbot.process_pdf(uploaded_file)
                    config.paper_content = text_content
                    st.session_state.file_content = f"The  contents：\n{text_content}"
                    st.toast("PDF uploaded! ", icon="💾")
                    
                    # Add 1st question
                    question_1 = EXTRACT_CASES_FROM_TEXT_PROMPT.format(text_content=text_content)

                    st.session_state.messages.append({
                        "role": "user",
                        "content": question_1, "timestamp": datetime.now().isoformat()
                    })
                    
                    # Get response for question A
                    response_1 = st.session_state.chatbot.get_response(st.session_state.messages)
                    st.session_state.messages.append({"role": "assistant", "content": response_1, "timestamp": datetime.now().isoformat()})

                    st.session_state.file_processed = True

                    # Chatbot ask the user to choose case and solver
                    if not st.session_state.ask_case_solver:
                        ask_to_choose_case_and_solver = ASK_TO_CHOOSE_CASE_AND_SOLVER
                        st.session_state.messages.append({
                            "role": "assistant", 
                            "content": ask_to_choose_case_and_solver,
                            "timestamp": datetime.now().isoformat()
                        })

                        st.session_state.ask_case_solver = True

    with st.sidebar:
        st.header("Upload the mesh file")
        uploaded_mesh_file = st.file_uploader(
            "Please upload mesh (only support the Fluent-format .msh)",
            type=['msh']
        )
        if uploaded_mesh_file:
            if not st.session_state.uploaded_grid:
                # Create save directory
                save_dir = pathlib.Path(config.TEMP_PATH)
                
                try:
                    # Build save path
                    file_path = save_dir / uploaded_mesh_file.name.replace(" ", "_")
                    
                    # Save uploaded file
                    with open(file_path, "wb") as f:
                        f.write(uploaded_mesh_file.getbuffer())
                    
                    st.toast(f"The mesh file has been saved: {file_path}", icon="💾")

                    config.case_grid = f"{config.TEMP_PATH}/{uploaded_mesh_file.name}"
                    config.path_cfg.grid_path = config.case_grid

                    config.case_boundaries = list(file_preparation.extract_boundary_names(file_path, config.grid_type)) # Extract boundary condition names

                    st.toast(f"The mesh file has been processed! ")

                    boundary_names = ", ".join(config.case_boundaries)
                    # print(config.case_boundaries)

                    config.case_boundary_names = boundary_names

                    info_after_mesh_processed = f'''You have uploaded a mesh file with boundary names as: {boundary_names}.\nNow the case are prepared and running in the background. Running information will be shown in the console.'''
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": info_after_mesh_processed,
                        "timestamp": datetime.now().isoformat()
                    })

                    st.session_state.ask_case_solver = True

                    st.session_state.uploaded_grid = True

                except Exception as e:
                    st.error(f"Failed at processed the mesh file: {str(e)}")              

    # Display conversation history
    if len(st.session_state.messages) > 0:
        for message in st.session_state.messages[1:]:
            if message["role"] == "user":
                st.chat_message("user").write(message["content"])
            else:
                if message["content"].startswith("Understand the user's answer"):
                    continue
                else:
                    st.chat_message("assistant").write(message["content"])

    if st.session_state.show_start == False:
        st.header('**Please upload the paper to start!**')
        st.session_state.show_start = True

    # User input
    if prompt := st.chat_input("Enter your requirement or reply."):
        
        st.chat_message("user").write(prompt)  # Display the user's original prompt in the UI

        if st.session_state.ask_case_solver and not st.session_state.user_answer_finished: # ask the user for Case_X, solver and turbulence

            guide_case_choose_prompt = GUIDE_CASE_CHOOSE_PROMPT.format(
                user_answer=prompt,
                json_response_sample=JSON_RESPONSE_SAMPLE,
                string_of_solver_keywords=config.string_of_solver_keywords,
                string_of_turbulence_model=config.string_of_turbulence_model
            )

            st.session_state.messages.append({"role": "user", "content": guide_case_choose_prompt, "timestamp": datetime.now().isoformat()})

            # Get assistant's response
            with st.chat_message("assistant"):
                max_retries = 3  # Maximum retry attempts
                for attempt in range(max_retries):
                    try:
                        response = st.session_state.chatbot.get_response(st.session_state.messages)
                        config.all_case_dict = json.loads(response)
                        break  # Parse successful, break loop
                    except json.JSONDecodeError:
                        print(f"Failed to get config.all_case_dict")
                        if attempt < max_retries - 1:
                            continue  # Continue retry
                        else:
                            raise  # Reached maximum retry attempts, raise exception

                qa = qa_modules.QA_NoContext_deepseek_R1()

                convert_json_to_md = f'''Convert the provided JSON string into a Markdown format where:
                    1. Each top-level JSON key becomes a main heading (#)
                    2. Its corresponding key-value pairs are rendered as unordered list items
                    3. Maintain the original key-value hierarchy in list format

                    The provided json string is as follow:{response}.
                '''

                md_form = qa.ask(convert_json_to_md)

                decorated_response = f'''You choose to simulate the cases with the following setups:\n{md_form}'''
                st.write(decorated_response)
                st.session_state.messages.append({"role": "assistant", "content": decorated_response, "timestamp": datetime.now().isoformat()})
                # later, fnae
                st.session_state.user_answer_finished = True

        else: 
            if st.session_state.user_answer_finished:
                modification_prompt = f'''Based on the user's new request: "{prompt}"
                
                Please modify the existing case configuration according to this request.
                The current configuration is: {json.dumps(config.all_case_dict, indent=2)}
                
                Generate an updated JSON with the same structure but incorporating the requested changes.
                Follow these requirements:
                1. Maintain the same JSON structure
                2. Ensure strict JSON syntax compliance
                3. Use double quotes for keys and string values
                4. Case_name must only contain [a-zA-Z0-9_]+ 
                5. Valid solvers: {config.string_of_solver_keywords}
                6. Valid turbulence models: {config.string_of_turbulence_model}
                
                Return ONLY the updated JSON content.
                '''
                st.session_state.messages.append({"role": "user", "content": modification_prompt, "timestamp": datetime.now().isoformat()})
                # Get modified configuration
                with st.chat_message("assistant"):
                    try:
                        response = st.session_state.chatbot.get_response(st.session_state.messages)
                        # Try to parse JSON response
                        updated_config = json.loads(response)
                        config.all_case_dict = updated_config
                        
                        # Convert to Markdown format for display
                        qa = qa_modules.QA_NoContext_deepseek_R1()
                        convert_json_to_md = f'''Convert the provided JSON string into a Markdown format where:
                            1. Each top-level JSON key becomes a main heading (#)
                            2. Its corresponding key-value pairs are rendered as unordered list items
                            3. Maintain the original key-value hierarchy in list format
                            
                            The provided json string is as follow:{response}.
                        '''
                        md_form = qa.ask(convert_json_to_md)
                        
                        # Display confirmation message
                        decorated_response = f'''Based on your request, I've updated the case configuration to:\n{md_form}'''
                        st.write(decorated_response)
                        # Add user-friendly response to chat history
                        st.session_state.messages.append({"role": "assistant", "content": decorated_response, "timestamp": datetime.now().isoformat()})
                    except json.JSONDecodeError:
                        st.write("I couldn't process that as a valid case modification. Please check your request and try again.")
                        st.session_state.messages.append({"role": "assistant", "content": "I couldn't process that as a valid case modification. Please check your request and try again.", "timestamp": datetime.now().isoformat()})
            else:
                # Original normal conversation processing logic
                st.session_state.messages.append({"role": "user", "content": prompt, "timestamp": datetime.now().isoformat()})
                # Get assistant's response
                with st.chat_message("assistant"):
                    response = st.session_state.chatbot.get_response(st.session_state.messages)
                    st.write(response)
                    st.session_state.messages.append({"role": "assistant", "content": response, "timestamp": datetime.now().isoformat()})

    if st.session_state.file_processed and st.session_state.user_answer_finished and not st.session_state.uploaded_grid:
        config.simulate_requirement = md_form

        st.write("If you don't have further requirement on the case setup. \n**Please upload the mesh of the Fluent .msh format.**")

    if st.session_state.uploaded_grid and st.session_state.file_processed and st.session_state.user_answer_finished:
        # read in preprocess OF tutorials
        print(f"**************** Preprocessing OF tutorials at {config.of_tutorial_dir} ****************")
        # if not config.flag_OF_tutorial_processed:
        #     preprocess_OF_tutorial.main()
        #     config.flag_OF_tutorial_processed = True
        preprocess_OF_tutorial.read_in_processed_merged_OF_cases()

        print("config.all_case_dict is:", config.all_case_dict)
        
        for key, value in config.all_case_dict.items():
            case_name = value["case_name"]
            print(f"***** start processing {key}: {case_name} *****")
            solver = value["solver"]

            try:
                turbulence_model = value["turbulence_model"]
            except KeyError:
                value["turbulence_model"] = None
                turbulence_model = None

            if turbulence_model not in config.turbulence_model_keywords:
                turbulence_model = None

            case_specific_description = value["case_specific_description"]

            other_physical_model = value["other_physical_model"]

            config.case_description = case_specific_description

            other_model_list = [
                "GRI", "TDAC", "LTS","common","Maxwell","Stokes"
            ]
            # Unified processing: whether input is string or list, convert to list
            if isinstance(other_physical_model, str):
                # String → single element list
                other_physical_model = [other_physical_model]
            elif not isinstance(other_physical_model, list):
                # Neither string nor list → empty list
                other_physical_model = []

            # Filter elements that exist in other_model_list
            other_physical_model = [m for m in other_physical_model if m in other_model_list]

            # If empty after filtering, set to None
            if not other_physical_model:
                other_physical_model = None

            # config.other_physical_model = other_physical_model
            # Load information to config.info
            config.case_info.case_name = case_name
            config.case_info.case_solver = solver
            config.case_info.turbulence_model = turbulence_model
            config.case_info.other_physical_model = other_physical_model
            config.case_info.case_description = case_specific_description

            main_run_chatcfd.run_case()


def main2(txt_file=""):
    # Initialize LLM
    print("==========================ChatCFD==============================")
    client = OpenAI(
            api_key=os.environ.get("DEEPSEEK_R1_KEY"),
            base_url=os.environ.get("DEEPSEEK_R1_BASE_URL")
        )
    
    # chatbot = qa_modules.QA_Context_deepseek_R1(SYSTEM_PROMPT)
    def basic_chat(messages):
        response= client.chat.completions.create(
            model=os.environ.get("DEEPSEEK_R1_MODEL_NAME"),
            messages=messages,
            temperature=0.9
        )
        return response.choices[0].message.content

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    
    # Read txt file content
    with open(txt_file, "r") as f:
        text_content = f.read()

    messages.append({"role": "user","content": EXTRACT_CASES_FROM_TEXT_PROMPT.format(text_content=text_content)})

    cases_in_text = basic_chat(messages)
    messages.append({"role": "assistant", "content": cases_in_text})

    print("\n**Chatbot:**\n", cases_in_text)

    ask_to_choose_case_and_solver = ASK_TO_CHOOSE_CASE_AND_SOLVER

    print("\n**Chatbot:**\n", ask_to_choose_case_and_solver)    

    if 0:
        user_answer = input("Please choose the case you want to simulate and the OpenFOAM solver you want to use: ")
    else:
        # Here you can use a preset answer for testing
        user_answer = "I want to simulate Case_1"

    guide_case_choose_prompt = GUIDE_CASE_CHOOSE_PROMPT.format(
        user_answer=user_answer,
        json_response_sample=JSON_RESPONSE_SAMPLE,
        string_of_solver_keywords=config.string_of_solver_keywords,
        string_of_turbulence_model=config.string_of_turbulence_model
    )

    messages.append({"role": "user", "content": guide_case_choose_prompt})
    get_all_case_dict = False
    max_retries = 3  # Set maximum retry attempts
    retry_count = 0
    while retry_count < max_retries and not get_all_case_dict:
        try:
            config.simulate_requirement = basic_chat(messages)

            print(f"**Chatbot:**\n {CONFIRM_SIMULATION_REQUIREMENT.format(simulate_requirement=config.simulate_requirement)}")

            config.all_case_dict = json.loads(config.simulate_requirement.lstrip("```json").rstrip("```").strip())
            get_all_case_dict = True
        except json.JSONDecodeError:
            print("Failed to parse JSON response:", config.simulate_requirement)
            retry_count += 1
            return

    print(f"**************** Preprocessing OF tutorials at {config.of_tutorial_dir} ****************")

    preprocess_OF_tutorial.read_in_processed_merged_OF_cases()

    for key, value in config.all_case_dict.items():
        case_name = value["case_name"]
        print(f"***** start processing {key}: {case_name} *****")
        solver = value["solver"]

        try:
            turbulence_model = value["turbulence_model"]
        except KeyError:
            value["turbulence_model"] = None
            turbulence_model = None

        if turbulence_model not in config.turbulence_model_keywords:
            value["turbulence_model"] = None
            turbulence_model = None

        case_specific_description = value["case_specific_description"]

        other_physical_model = value["other_physical_model"]

        config.case_description = case_specific_description

        other_model_list = [
            "GRI", "TDAC", "LTS","common","Maxwell","Stokes"
        ]

        # Unified processing: whether input is string or list, convert to list
        if isinstance(other_physical_model, str):
            # String → single element list
            other_physical_model = [other_physical_model]
        elif not isinstance(other_physical_model, list):
            # Neither string nor list → empty list
            other_physical_model = []

        # Filter elements that exist in other_model_list
        other_physical_model = [m for m in other_physical_model if m in other_model_list]

        # If empty after filtering, set to None
        if not other_physical_model:
            other_physical_model = None

        # Load information to config.info
        config.case_info.case_name = case_name
        config.case_info.case_solver = solver
        config.case_info.turbulence_model = turbulence_model
        config.case_info.other_physical_model = other_physical_model
        config.case_info.case_description = case_specific_description

        main_run_chatcfd.run_case()


if __name__ == "__main__":

    # ======= **0. You need to make the settings according to the requirements here** =======
    config.mode = 0    # 0 or 1, corresponding to using streamlit or python for startup
    config.grid_type = "polyMesh"   # msh or polyMesh
    config.run_cfg.run_time = 1

    # ======= 1. Program startup settings =======
    if config.mode == 0:  # 0: With frontend, streamlit
        # streamlit run src/chatbot.py --server.port [your port setting, such as 8502]

        config.grid_type = "msh"
        # print("Please use the following command to start the program:")
        # print("  streamlit run src/chatbot.py --server.port [your port setting, such as 8502]")
        # print("Please enter the PDF, mesh file (.msh format), and simulation requirements in the interactive interface...")
    
        main()  # streamlit run src/chatbot.py --server.port 8502

    elif config.mode == 1:  # 1: No frontend, run directly
        # python foambench_main.py --openfoam_path $WM_PROJECT_DIR --output ./output --prompt_path "./user_requirement.txt"

        parser = argparse.ArgumentParser(description="ChatCFD: AI-Driven CFD Simulation Setup and Execution")

        parser.add_argument("--case_description_path", type=str, default=os.path.join(config.path_cfg.root_dir, "pdf/counterFlowFlame2D.txt"), help="Path to the PDF or txt file")  # , required=True
        parser.add_argument("--grid_path", type=str, default=os.path.join(config.path_cfg.root_dir, "grids/combustion_reactingFoam_laminar_counterFlowFlame2D/constant/polyMesh"), help="Path to the grid file (.msh or polyMesh format)")  # , required=True
        parser.add_argument("--run_time", type=int, default=config.run_cfg.run_time, help="number of simulation runs")

        args = parser.parse_args()

        config.path_cfg.case_description_path = args.case_description_path
        config.path_cfg.grid_path = args.grid_path
        config.run_cfg.run_time = args.run_time

        print("Your config:")
        print(f"  config.path_cfg.case_description_path={config.path_cfg.case_description_path}")
        print(f"  config.path_cfg.grid_path={config.path_cfg.grid_path}")
        print(f"  config.run_cfg.run_time={config.run_cfg.run_time}")

        config.case_info.file_name = os.path.basename(config.path_cfg.case_description_path).rstrip("txt").rstrip("pdf").rstrip(".")
        config.path_cfg.output_path = os.path.join(config.path_cfg.output_dir, config.case_info.file_name)

        case_stat_path = os.path.join(config.path_cfg.output_dir, "case_stat.txt")
        try:
            main2(config.path_cfg.case_description_path)

            with open(case_stat_path, "a") as f:
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                f.write("="*50 + "\n")
                f.write(f"[{timestamp}] {config.path_cfg.case_description_path}: success.\n")
            print(f"Success cases: {config.path_cfg.case_description_path}")
        
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"\nError processing {config.path_cfg.case_description_path}: {e}")

            with open(case_stat_path, "a") as f:
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                f.write("="*50 + "\n")
                f.write(f"[{timestamp}] {config.path_cfg.case_description_path}: failure.\n")
                f.write(f"Error details: \n{str(e)}\n")
            print(f"Error cases: {config.path_cfg.case_description_path}")
    else:
        sys.exit("Error: mode setting shall be 0 or 1")
