# ChatCFD: An LLM-Driven Agent for End-to-End CFD Automation with Structured Knowledge and Reasoning

## ✨ Quick Introduction

### Overview

Computational Fluid Dynamics (CFD) is widely constrained by its operational complexity, high demand for specialized knowledge, and steep learning curve. ChatCFD is an advanced Large Language Model (LLM)-driven agent system designed to solve these challenges. Users can now complete complex OpenFOAM case configuration and execution using natural language or descriptions from academic papers, all through a conversational chat interface. This significantly simplifies the complex simulation configuration and execution processes, making CFD simulations more accessible to users lacking extensive domain experience, ultimately achieving end-to-end automation of the CFD workflow.

ChatCFD operates within the popular open-source CFD framework, OpenFOAM.

For ChatCFD's system architecture, performance benchmarks, and detailed technology, please refer to our preprint paper published on arXiv:

  - [ChatCFD: An LLM-Driven Agent for End-to-End CFD Automation with Structured Knowledge and Reasoning](https://arxiv.org/abs/2506.02019v2)

To support academic research and community development, we have open-sourced the ChatCFD test dataset on Hugging Face:

  - 🤗 Dataset Repository: [ConMoo/ChatCFD_dataset](https://huggingface.co/datasets/ConMoo/ChatCFD_dataset/tree/main)

### Project Highlights

ChatCFD adopts a multi-agent architecture based on DeepSeek-R1 and DeepSeek-V3, combined with the following key technologies to ensure a high success rate and physical accuracy:

- **Domain-Specific Structured Knowledge Base:** Ensures the accuracy of CFD concepts and configurations.

- **Iterative Reflection Mechanism:** Automatically detects and fixes potential errors in OpenFOAM configurations.

- **Natural Language to Case Conversion Capability:** Translates complex academic descriptions into executable OpenFOAM cases.

In terms of user experience, ChatCFD employs an Interactive Multimodal Interface: Providing a smooth, ChatGPT-like conversational experience, supporting the uploading of PDF papers, mesh files, and natural language instructions for comprehensive input support.

### Key Function: One-Click from Natural Language (short sentences, papers, technical manuals, etc.) to Simulation

ChatCFD provides services through a conversational chat interface or command-line instructions, enabling users to launch CFD simulations with unprecedented convenience:

1. **Input Description:** Users can specify the required CFD case through a natural language description or by uploading academic papers, technical manuals, etc., in PDF/TXT format.

2. **Mesh Provision:** Users provide the corresponding mesh file (supporting OpenFOAM or Fluent formats).

3. **Automatic Configuration:** The ChatCFD agent system automatically identifies and interprets the CFD case and all its parameter settings described in the paper, and automatically completes all OpenFOAM case configurations and simulation setups.

This enables users who lack extensive CFD domain expertise to perform CFD simulations easily and accurately.

![ChatCFD Overview](figures/fig1.illustration2.png)

## 📖 Table of Contents
- [ChatCFD: An LLM-Driven Agent for End-to-End CFD Automation with Structured Knowledge and Reasoning](#chatcfd-an-llm-driven-agent-for-end-to-end-cfd-automation-with-structured-knowledge-and-reasoning)
  - [Quick Introduction](#-quick-introduction)
    - [Overview](#overview)
    - [Project Highlights](#project-highlights)
    - [Key Function](#key-function-one-click-from-natural-language-short-sentences-papers-technical-manuals-etc-to-simulation)
  - [Table of Contents](#-table-of-contents)
  - [System Requirements](#️-system-requirements)
      - [Core Dependencies](#core-dependencies)
      - [Python Dependencies Overview](#python-dependencies-overview)
  - [Installation](#-installation-guide)
    - [Step 1. Clone the Repository](#step-1-clone-the-repository)
    - [Step 2. Configure Python Environment](#step-2-configure-python-environment)
    - [Step 3. Verify Key Components](#step-3-verify-key-components)
    - [Step 4. Download Sentence Transformer Model](#step-4-download-sentence-transformer-model)
    - [Step 5. Install OpenFOAM2406](#step-5-install-openfoam2406)
    - [Step 6. Set up Configuration File](#step-6-set-up-configuration-file)
  - [Running ChatCFD](#-running-chatcfd)
    - [Mode 0: Web Frontend Interface (Recommended)](#mode-0-web-frontend-interface-recommended)
    - [Mode 1: Command Line Launch (Batch Processing)](#mode-1-command-line-launch-batch-processing)
  - [File Structure Description](#-file-structure-description)
  - [Performance Metrics](#-performance-metrics)
  - [Community & Contact](#-community--contact)
  - [Project Status](#-project-status)
  - [Citation](#citation)

## ⚙️ System Requirements

### Core Dependencies
- Python 3.11 or higher, but it is best to avoid Python 3.12 as this version may cause unknown issues when using PyFoam during direct execution.
- OpenFOAM v2406. If other versions of OpenFOAM are used, files like `processed_merged_OF_cases.json` in `database_OFv24` may become incompatible, affecting ChatCFD's performance.
- CUDA-enabled GPU (Optional, but recommended for better performance).

| Dependency | Version/Requirement | Notes |
|---------|---------|---------|
| Operating System | Linux / Windows (WSL2) | Linux is recommended to ensure OpenFOAM compatibility. |
| CFD Simulation Software | **OpenFOAM v2406** | ⚠️ Recommended version is v2406. Using other versions will lead to a mismatch with pre-built knowledge base files (e.g., `database_OFv24/*.json`), severely impacting ChatCFD's performance and accuracy. |
| Python | Python 3.11.4 | ⚠️ Please avoid Python 3.12, as this version may have unknown conflicts with the PyFoam library. |
| Hardware | CUDA-enabled GPU | Optional, but recommended to accelerate the running of some components. |

### Python Dependencies Overview
All dependencies are included in the `chatcfd_env.yml` file, covering the following main areas:
- **Machine Learning & AI**:
  - PyTorch 2.6.0
  - Transformers 4.50.3
  - Sentence-Transformers 4.0.1
  - FAISS-CPU 1.7.4 (for vector similarity search)
  - Scikit-learn 1.6.1
  - NumPy 1.26.4
  - Pandas 2.2.3

- **Web & API**:
  - Streamlit 1.41.1 (for web interface)
  - OpenAI 1.39.0
  - LangChain 0.1.19
  - FastAPI and related dependencies

- **PDF Processing**:
  - PDFPlumber 0.11.5
  - PyPDF2 3.0.1
  - PDFMiner.six 20231228

- **OpenFOAM Integration**:
  - PyFoam 2023.7

## 🚀 Installation Guide

### Step 1. Clone the Repository
First, clone the ChatCFD repository locally and navigate into the project directory.

```bash
git clone https://github.com/ConMoo/ChatCFD.git

cd ChatCFD
```

### Step 2. Configure Python Environment
Use conda to create and activate the virtual environment to install all necessary Python dependencies.

```Bash
# 1. Create and install the environment
conda env create -f chatcfd_env.yml

# 2. Activate the environment
conda activate chatcfd
```
If `conda env create -f chatcfd_env.yml` fails to configure the virtual environment, you can try the following alternative configuration method:
```Bash
conda create -n chatcfd python=3.11.4

conda activate chatcfd

pip install -r requirements.txt   # May require installing additional packages depending on the execution environment
```
### Step 3. Verify Key Components

Verify that the key dependencies in the environment are installed correctly.

```Bash
# Verify FAISS (Vector Search)
python -c "import faiss; print(faiss.IndexFlatL2(10))"

# Verify PyFoam (OpenFOAM Interface)
python -c "from PyFoam.RunDictionary.ParsedParameterFile import ParsedParameterFile; print('PyFoam OK')"
```
### Step 4. Download Sentence Transformer Model
Download the embedding model all-mpnet-base-v2 required by ChatCFD.

```Bash

# 1. Download model files
python test_env/download_model.py

# 2. Run test (Optional)
python test_env/test_all_mpnet_base-v2.py
```
Expected Output Example (CPU Environment):
```
GPU Available: False
GPU Name: None
Similarity 0-1: 0.383
Similarity 0-2: 0.182
```
Note: Results will differ if using GPU.

### Step 5. Install OpenFOAM2406
Install the specified version through the official OpenFOAM Debian repository.

```Bash
# 1. Add repository
curl -s [https://dl.openfoam.com/add-debian-repo.sh](https://dl.openfoam.com/add-debian-repo.sh) | sudo bash

# 2. Update and install OpenFOAM 2406
sudo apt-get update
sudo apt-get install openfoam2406-default

# 3. Import OpenFOAM environment variables
export WM_PROJECT_DIR=/usr/lib/openfoam/openfoam2406
```
Manual download link: https://develop.openfoam.com/Development/openfoam/-/wikis/precompiled/debian

### Step 6. Set up Configuration File
Edit the configuration file `inputs/chatcfd_config.json`, replacing the placeholder `[[[...]]]` for API keys and URLs.

If you installed OpenFOAM according to Step 4, `OpenFOAM_path` and `OpenFOAM_tutorial_path` usually do not need modification. If the model download in Step 5 failed and you downloaded the model manually, you will need to set `sentence_transformer_path`.

```JSON
{
    "DEEPSEEK_V3_KEY" : "[[[you_API_key]]]",
    "DEEPSEEK_V3_BASE_URL" : "[[[API_URL]]]",
    "DEEPSEEK_V3_MODEL_NAME" : "deepseek-v3-250324",
    "V3_temperature":0.7,
    "DEEPSEEK_R1_KEY" : "[[[you_API_key]]]",
    "DEEPSEEK_R1_BASE_URL" : "[[[API_URL]]]",
    "DEEPSEEK_R1_MODEL_NAME" : "deepseek-r1-250528",
    "R1_temperature":0.9,
    "run_time":1,
    "OpenFOAM_path":"/usr/lib/openfoam/openfoam2406",
    "OpenFOAM_tutorial_path":"/usr/lib/openfoam/openfoam2406/tutorials",
    "max_running_test_round":30,
    "pdf_chunk_d" : 1.5,
    "sentence_transformer_path": ""
}
```

## 🏃 Running ChatCFD
### Mode 0: Web Frontend Interface (Recommended)
Launch the interactive chat interface using Streamlit.

```Bash

streamlit run src/chatbot.py --server.port "[[[your_port_setting, e.g., 8501]]]"
```
After running, access http://localhost:[port] in your browser. In the interface, users can upload paper PDFs, specify the CFD case, and upload mesh files to start the simulation as prompted.

**Specific operational steps:**
1. Input Case Description Document:
    - In the `Upload the Documents` area, upload the description file containing the detailed CFD simulation information (e.g., academic papers, natural language instructions, etc., typically in PDF format).

2. Intelligent Identification and Configuration Review:

    - ChatCFD will automatically parse the description file to identify the executable CFD simulation cases and their corresponding parameter settings.

    - After the user selects the specific case they wish to execute, the system will output a summary of the basic configuration.

    - The user can then make interactive modifications and confirmations to fundamental settings such as the solver, turbulence model, and boundary conditions via the chat interface.

3. Provide Mesh and Start Simulation:

    - Once the case configuration is finalized, upload the corresponding computational mesh file (e.g., .msh format) in the `Upload the mesh file` area.

    - The system will automatically complete the full configuration and file generation for the OpenFOAM case, initiating the end-to-end automated CFD simulation.

4. Result Storage:

    - Upon simulation completion, the final CFD results and logs will be automatically saved to the `run_chatcfd` folder in the project root directory


Web Interface Upon Launch:

![Web](figures/web.png)

### Mode 1: Command Line Launch (Batch Processing)
Specify input files directly via command line arguments to start the simulation.

```Bash
python src/chatbot.py --document_path "<path_to_pdf_or_txt>" --grid_path "<path_to_mesh_file>" --run_time "<int_number_of_iterations>"
```
**Specific operational steps:**
1. Prepare Input: 
    - Ensure that the command-line arguments (`--document_path`, `--grid_path`, and `--run_time`) are correctly specified.

2. Execute Calculation: 
    - Run the command directly. ChatCFD will automatically complete the entire process of case identification, configuration, execution, and error correction.

3. Retrieve Results: 
    - The final CFD simulation results and logs will be saved in the `run_chatcfd` folder located in the project root directory.


## 📁 File Structure Description

The root directory of the project, `ChatCFD/`, is structured as follows, mainly divided into configuration, data, code, testing, and output sections. Below is the detailed file tree structure:
```
ChatCFD/
├── database_OFv24/                         # OpenFOAM v2406 Database and Configuration
│   ├── final_OF_solver_required_files.json # List of files required by the solver
│   ├── final_OF_turbulence_required_files.json # List of files required by the turbulence model
│   ├── OF_bc_entry.json                    # Boundary condition entries
│   ├── OF_case_dimensions.json             # Case dimension information
│   └── processed_merged_OF_cases.json      # Processed and merged case data
├── grids/                                  # Mesh File Directory
│   ├── naca0012.msh                        # NACA0012 airfoil mesh file
│   └── Yu_2023_nozzle.msh                  # Nozzle mesh file
├── inputs/                                 # Input Configuration Files
│   └── chatcfd_config.json                 # Configuration file, includes API keys, paths, etc.
├── pdf/                                    # PDF File Directory, example papers or case descriptions
│   ├── sun_2023_naca0012.pdf               # Literature describing the NACA0012 case simulation
│   └── Yu_2023_nozzle.pdf                  # Literature describing the Nozzle case simulation
├── run_chatcfd/                            # Run Directory, stores generated OpenFOAM cases
│   └── sample_NACA0012_AOA10_kOmegaSST/    # Simulation results for the 2D incompressible NACA0012 case generated by ChatCFD
├── src/                                    # Source Code Directory
│   ├── chatbot.py                          # Main chatbot script, implements the conversational interface and user interaction
│   ├── file_preparation.py                 # File preparation module, responsible for generating and configuring OpenFOAM case files
│   ├── main_run_chatcfd.py                 # Main run script, coordinates the entire CFD automation workflow.
│   ├── preprocess_OF_tutorial.py           # OpenFOAM tutorial pre-processing script, used for initial parsing and extraction of tutorial data
│   ├── qa_modules.py                       # QA module, integrates LLM for reasoning and configuration, and performs token consumption statistics
│   ├── run_of_case.py                      # OpenFOAM case run script, executes the simulation.
│   ├── file_corrector.py                   # Case file correction module, automatically detects and fixes configuration errors
│   └── config.py                           # Configuration file module, manages system settings and paths
├── test_env/                               # Environment Test Scripts
│   ├── download_model.py                   # Script to download the Sentence Transformer model
│   └── config.py                           # Test whether the all-mpnet-base-v2 model can correctly calculate similarity
└── utils/                                  # Utility Script Directory, auxiliary functions such as data processing
```
![FrameWork](figures/fig2.corescheme.png)

## 📊 Performance Metrics
ChatCFD demonstrates exceptional automation capabilities in both basic OpenFOAM cases and the reproduction of actual academic papers, with superior robustness, accuracy, and cost-efficiency compared to existing systems.

- **OpenFOAM Basic Test (315 basic cases):** Comprehensive test results for 315 basic cases extracted from the official OpenFOAM tutorials:

  - **Execution Success Rate (run):** 82%

  - **Physical Fidelity (phy√):** 59%

  - **Average Case Cost:** $0.20

- **Complex Literature Reproduction Capability:** We tested complex CFD cases extracted from academic papers, demonstrating ChatCFD's ability to translate natural language descriptions into complex configurations:

  - **Incompressible NACA0012 Case Success Rate:** 40%

  - **Compressible Nozzle Case Success Rate:** 30%

  - **Case Cost:** $0.30

- **Model Flexibility and Intelligence:**

  - **Adaptive Solver Selection:** Can intelligently match and recommend the most suitable OpenFOAM solver based on user-specified simulation requirements (e.g., steady-state/transient, incompressible/compressible).

  - **Natural Language Model Switching:** Users can easily replace or configure different turbulence models (e.g., k-epsilon, Spalart-Allmaras, etc.) for existing cases simply through natural language instructions.

Through its excellent execution success rate and physical fidelity, ChatCFD provides the CFD community with a reliable, efficient, and low-cost end-to-end automation solution.

## 💬 Community & Contact
If you encounter any issues, discover a bug, or have feature suggestions during the configuration, running, or usage of ChatCFD, we highly welcome you to contact and communicate with us through the following methods:

**Submit Issues:** Raise your questions on the Issues page of this GitHub repository.

**Email:** You can also directly contact the core members of the project team via email, for example, 454114084@qq.com, hukang1scu@gmail.com

We are committed to maintaining and improving the project and are happy to provide you with support.

## ⏳ Project Status
**Status:** Active Development & Maintenance

This project is constantly being updated and optimized. We regularly push new features, performance improvements, and knowledge base updates. We welcome you to keep following us.

## Citation

If you use the ChatCFD system or related data in your research, please cite our preprint paper:

```
@article{fan2026chatcfd,
  title={ChatCFD: A Large Language Model-Driven Agent for End-to-End Computational Fluid Dynamics Automation with Structured Knowledge and Reasoning},
  author={Fan, E and Hu, Kang and Wu, Zhuowen and Ge, Jiangyang and Miao, Jiawei and Zhang, Yuzhi and Sun, He and Wang, Weizong and Zhang, Tianhan},
  journal={Advanced Intelligent Discovery},
  pages={e202500174},
  year={2026},
  publisher={Wiley Online Library}
}
```
