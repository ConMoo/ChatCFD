import os
import sys
import yaml
import json
import subprocess

from dataclasses import dataclass, field, fields

# Configuration settings for ChatCFD
@dataclass
class path_config:
    """Path configuration"""
    # Basic program path settings
    root_dir: str = field(default="", metadata={"description": "Program root directory"})
    src_dir: str = field(default=os.path.dirname(os.path.abspath(__file__)), metadata={"description": "Source code directory"})
    output_dir: str = field(default="", metadata={"description": "Output directory"})
    database_dir: str = field(default="", metadata={"description": "Database directory"})
    temp_dir: str = field(default="", metadata={"description": "Temporary directory for storing uploaded PDF, msh files, etc."})

    json_config_path: str = field(default="", metadata={"description": "Configuration file path"})
    
    case_description_path: str = field(default="", metadata={"description": "Path to PDF or txt file describing simulation case"})    # Will replace config.pdf_path later
    grid_path: str = field(default="", metadata={"description": "Path to mesh file in msh or polyMesh format"})       # replace config.case_grid later
    output_path: str = field(default="", metadata={"description": "OpenFOAM case output path under output_dir, same name as case description file"})
    output_case_path:str = field(default="", metadata={"description": "OpenFOAM case output path under output_path, specific name set by LLM"})

    def __post_init__(self):
        """Automatically set dependent fields after initialization"""
        if not self.root_dir:
            self.root_dir = os.path.dirname(self.src_dir)
        if not self.output_dir:
            self.output_dir = os.path.join(self.root_dir, "run_chatcfd")  # output_dir/output_path/output_case_path
        if not self.database_dir:
            self.database_dir = os.path.join(self.root_dir, "database_OFv24")
        if not self.temp_dir:
            self.temp_dir = os.path.join(self.root_dir, "temp")
        if not self.json_config_path:
            self.json_config_path = os.path.join(self.root_dir, "inputs", "chatcfd_config.json")

@dataclass
class dependencies_config:
    # Software and model path settings
    OpenFOAM_path: str = field(default="/usr/lib/openfoam/openfoam2406", metadata={"description": "OpenFOAM installation path"})
    OpenFOAM_tutorials_path: str = field(default="", metadata={"description": "OpenFOAM tutorials path"})
    sentence_transformer_path: str = field(default="/home/all-mpnet-base-v2", metadata={"description": "Sentence Transformer model path"})
    
    def __post_init__(self):
        """Automatically set dependent fields after initialization"""
        if not self.OpenFOAM_tutorials_path:
            self.OpenFOAM_tutorials_path = os.path.join(self.OpenFOAM_path, "tutorials")

@dataclass
class llm_config:
    """LLM configuration"""
    # R1 model configuration
    DEEPSEEK_R1_KEY : str = field(default="", metadata={"description": "DeepSeek-R1 API Key"})
    DEEPSEEK_R1_BASE_URL : str = field(default="https://ark.cn-beijing.volces.com/api/v3", metadata={"description": "DeepSeek-R1 API Base URL"})
    R1_temperature: float = field(default=0.9, metadata={"description": "R1 model temperature"})

    # V3 model configuration
    DEEPSEEK_V3_KEY : str = field(default="", metadata={"description": "DeepSeek-V3 API Key"})
    DEEPSEEK_V3_BASE_URL : str = field(default="https://ark.cn-beijing.volces.com/api/v3", metadata={"description": "DeepSeek-V3 API Base URL"})
    V3_temperature: float = field(default=0.7, metadata={"description": "V3 model temperature"})

@dataclass
class run_config:
    """Runtime configuration"""
    mode: int = field(default=0, metadata={"description": "Runtime mode, 0: with frontend, run after frontend uploads PDF and mesh files, 1: without frontend, run after setting PDF or txt and mesh file paths"})
    grid_type: str = field(default="msh", metadata={"description": "Mesh file type, msh or polyMesh"})

    run_time: int = field(default=3, metadata={"description": "Number of runs for a single case"})
    max_running_test_round: int = field(default=30, metadata={"description": "Maximum reflection iteration rounds"})

@dataclass
class pdf_config:
    """PDF processing configuration"""
    pdf_chunk_d: float = field(default=1.5, metadata={"description": "Distance threshold for relevance analysis"})
    
    pdf_content: str = field(default="", metadata={"description": "PDF content, text content obtained after processing"})

    pass


class ConfigManager:
    """Configuration manager"""
    def __init__(self, case_des_file_path="", grid_path=""):
        """Initialize configuration manager"""
        self.path_config = path_config(
            case_description_path=case_des_file_path,
            grid_path=grid_path,
            output_path=os.path.join(path_config.output_dir, os.path.basename(case_des_file_path).rstrip('txt').rstrip('pdf').rstrip('.'))
            )
        self.dependencies_config = dependencies_config()
        self.llm_config = llm_config()
        self.run_config = run_config()
        self.pdf_config = pdf_config()

    def _load_config_from_json(self):
        """Load configuration from ChatCFD/inputs/chatcfd_config.json"""
        print("Loading configuration from JSON...")
        if not os.path.exists(self.path_config.json_config_path):
            print(f"Configuration file {self.path_config.json_config_path} does not exist, using default settings")

        try:
            with open(self.path_config.json_config_path, 'r', encoding='utf-8') as f:
                config_content = json.load(f)

            # Set path_config
            valid_fields = {field.name for field in fields(dependencies_config)}  # Get all field names of dependencies_config
            filtered_dict = {k: v for k, v in config_content.items() if k in valid_fields}  # Filter invalid fields
            dependencies_cfg = dependencies_config(**filtered_dict)

            # Set llm_config
            valid_fields = {field.name for field in fields(llm_config)}  # Get all field names of llm_config
            filtered_dict = {k: v for k, v in config_content.items() if k in valid_fields}  # Filter invalid fields
            llm_cfg = llm_config(**filtered_dict)
            
            # Set run_config
            valid_fields = {field.name for field in fields(run_config)}  # Get all field names of run_config
            filtered_dict = {k: v for k, v in config_content.items() if k in valid_fields}  # Filter invalid fields
            run_cfg = run_config(**filtered_dict)

            # Set pdf_config
            valid_fields = {field.name for field in fields(pdf_config)}  # Get pdf_config field names
            filtered_dict = {k: v for k, v in config_content.items() if k in valid_fields}  # Filter invalid fields
            pdf_cfg = pdf_config(**filtered_dict)

        except:
            import traceback
            traceback.print_exc()
            print(f"Error in configuration file {self.path_config.json_config_path}, using default settings")
            dependencies_cfg = dependencies_config()
            llm_cfg = llm_config()
            run_cfg = run_config()
            pdf_cfg = pdf_config()
        return dependencies_cfg, llm_cfg, run_cfg, pdf_cfg

    def _ensure_directory_exists(self):
        """Ensure all necessary directories exist"""
        directories = [
            self.path_config.database_dir,
            self.path_config.temp_dir
        ]
        for directory in directories:
            if not os.path.exists(directory):
                os.makedirs(directory)
                print(f"Directory '{directory}' was missing and has been automatically created")

        if not os.path.exists(self.path_config.output_dir):
            os.makedirs(self.path_config.output_dir)
            print(f"Created output directory '{self.path_config.output_dir}' for storing run results")

    def load_config(self):
        """Load configuration"""
        dependencies_cfg, llm_cfg, run_cfg, pdf_cfg = self._load_config_from_json()
        self.dependencies_cfg = dependencies_cfg
        self.llm_config = llm_cfg
        self.run_config = run_cfg
        self.pdf_config = pdf_cfg

        os.environ["DEEPSEEK_V3_KEY"] = self.llm_config.DEEPSEEK_V3_KEY
        os.environ["DEEPSEEK_V3_BASE_URL"] = self.llm_config.DEEPSEEK_V3_BASE_URL
        os.environ["DEEPSEEK_V3_MODEL_NAME"] = getattr(self.llm_config, 'DEEPSEEK_V3_MODEL_NAME', 'deepseek-v3-250324')

        os.environ["DEEPSEEK_R1_KEY"] = self.llm_config.DEEPSEEK_R1_KEY
        os.environ["DEEPSEEK_R1_BASE_URL"] = self.llm_config.DEEPSEEK_R1_BASE_URL
        os.environ["DEEPSEEK_R1_MODEL_NAME"] = getattr(self.llm_config, 'DEEPSEEK_R1_MODEL_NAME', 'deepseek-r1-250528')

        self._ensure_directory_exists()  # Ensure all necessary directories exist

    def save_config_to_json(self):
        """Save configuration to JSON file"""
        print("----- Saving configuration to JSON -----")
        config_data = {
            "OpenFOAM_path": self.dependencies_config.OpenFOAM_path,
            "OpenFOAM_tutorials_path": self.dependencies_config.OpenFOAM_tutorials_path,
            "sentence_transformer_path": self.dependencies_config.sentence_transformer_path,
            "DEEPSEEK_R1_KEY": self.llm_config.DEEPSEEK_R1_KEY,
            "DEEPSEEK_R1_BASE_URL": self.llm_config.DEEPSEEK_R1_BASE_URL,
            "R1_temperature": self.llm_config.R1_temperature,
            "DEEPSEEK_V3_KEY": self.llm_config.DEEPSEEK_V3_KEY,
            "DEEPSEEK_V3_BASE_URL": self.llm_config.DEEPSEEK_V3_BASE_URL,
            "V3_temperature": self.llm_config.V3_temperature,
            "run_time": self.run_config.run_time,
            "max_running_test_round": self.run_config.max_running_test_round,
            "pdf_chunk_d": self.pdf_config.pdf_chunk_d,
            "mode": self.run_config.mode,               # This setting is not in normal json_config_path
            "grid_type": self.run_config.grid_type,     # This setting is not in normal json_config_path
        }
        with open(self.path_config.json_config_path, 'w', encoding='utf-8') as f:
            json.dump(config_data, f, indent=4, ensure_ascii=False)
        print("----- Configuration Saved -----")

    def load_openfoam_env(self):
        """Load OpenFOAM environment variables and local machine environment variables into current Python process"""
        print("Loading openfoam enviroment...")
        try:
            command =  f'source {self.dependencies_cfg.OpenFOAM_path}/etc/bashrc && env'
            output = subprocess.run(
                command,
                shell=True,
                executable="/usr/bin/bash",  # Ensure using Bash
                check=True,  # Check if command succeeds
            text=True,
            capture_output=True,
            )
            # Inject environment variables
            for line in output.stdout.splitlines():
                if "=" in line:
                    key, value = line.split("=", 1)
                    os.environ[key] = value
        except subprocess.CalledProcessError as e:
            print(f"Please check if load_config() was used correctly before running this function")
            print(f"Failed to load OpenFOAM environment: {e.stderr}")
            raise


# Create global configuration manager instance for ChatCFD related configurations
config_manager = ConfigManager()  # Create global configuration manager instance
config_manager.load_config()      # Load configuration
config_manager.load_openfoam_env()  # Load OpenFOAM environment variables

llm_cfg = config_manager.llm_config
dependencies_cfg = config_manager.dependencies_config
run_cfg = config_manager.run_config
pdf_cfg = config_manager.pdf_config
path_cfg = config_manager.path_config

# input_file_name = os.path.basename(path_cfg.case_description_path).rstrip("txt").rstrip("pdf").rstrip(".")  # 用于储存用户输入的文件名

# Store case-related configurations, status, content, etc.
@dataclass
class case_status:
    file_name: str = field(default="", metadata={"description": "Case description file name"})
    case_description: str = field(default="", metadata={"description": "Case description"})
    case_name: str = field(default="", metadata={"description": "Case name"})
    case_solver: str = field(default="", metadata={"description": "Case solver"})
    turbulence_model: str = field(default="", metadata={"description": "Turbulence model"})
    other_physical_model: str = field(default="", metadata={"description": "Other physical models"})
    file_structure: list = field(default_factory=list, metadata={"description": "File structure"})
    reference_file_name: str = field(default="", metadata={"description": "Main reference file name when generating initial file structure and content"})
# Store mesh-related configurations, status, content, etc.
@dataclass
class grid_status:
    # grid_path: str = field(default=path_cfg.grid_path, metadata={"description": "Mesh file path"})
    # grid_type: str = field(default=run_cfg.grid_type, metadata={"description": "Mesh file type, msh or polyMesh"})
    mesh_convert_success: bool = field(default=False, metadata={"description": "Whether mesh conversion succeeded"})

    grid_boundary_conditions: dict = field(default_factory=dict, metadata={"description": "Mesh boundary conditions, key is boundary name, value is mesh boundary condition"})
    grid_boundary_init: str = field(default="", metadata={"description": "Initial boundary file content"})
    field_ic_bc_from_input: dict = field(default_factory=dict, metadata={"description": "Field file boundary and initial conditions, key is field file name, value is boundary_name->bc_ic"})

case_info = case_status()  # Case-related information
grid_info = grid_status()  # Mesh-related information


# @dataclass
# class correct_trajectory:
error_history = []
correct_trajectory = []  # Store modified file content, including original and modified files {file_name: [original_content, modified_content]}, no original_content for new files


mode = 0    # 0: With frontend, accepts mesh files and runs normally; 1: Without frontend, runs directly
grid_type = "msh"   # Mesh file format, either 'msh' or 'polyMesh'


of_path = dependencies_cfg.OpenFOAM_path
sentence_transformer_path = dependencies_cfg.sentence_transformer_path


def ensure_directory_exists(directory_path):
    # Check if the directory exists
    if not os.path.exists(directory_path):
        # If it doesn't exist, create the directory
        os.makedirs(directory_path)
        print(f"Directory '{directory_path}' has been created.")

SRC_DIR = "src"
Src_PATH = os.path.dirname(os.path.abspath(__file__))
Base_PATH = os.path.dirname(Src_PATH)

with open(f"{Base_PATH}/inputs/chatcfd_config.json", "r", encoding="utf-8") as f:
    config_data = json.load(f)

of_path = config_data["OpenFOAM_path"]
sentence_transformer_path = config_data["sentence_transformer_path"]

R1_temperature = 0.9
V3_temperature = 0.7

all_case_requirement_json = None

all_case_dict = None        # Case summary generated by LLM based on PDF

case_description = None     # Case description

other_physical_model = None # str or list

target_case_requirement_json = None

def convert_boundary_names_to_lowercase(data):
    if isinstance(data, dict):
        new_dict = {}
        for key, value in data.items():
            if key == "boundaries":
                new_dict[key] = {k.lower(): v for k, v in value.items()}
            else:
                new_dict[key] = convert_boundary_names_to_lowercase(value)
        return new_dict
    elif isinstance(data, list):
        return [convert_boundary_names_to_lowercase(item) for item in data]
    else:
        return data

global_target_case_dict = None

case_boundaries = []            # Case boundary conditions
case_solver = None              # Case solver
case_turbulence_type = "laminar"
case_turbulence_model = "invalid"
case_boundary_names = None

reference_case_searching_round = 10

case_name = None

Database_OFv24_PATH = f'{Base_PATH}/database_OFv24'
TEMP_PATH = f'{Base_PATH}/temp'
OUTPUT_CHATCFD_PATH = f'{Base_PATH}/run_chatcfd'
OUTPUT_PATH = None

case_grid = None
pdf_path = None

ensure_directory_exists(Database_OFv24_PATH)
ensure_directory_exists(OUTPUT_CHATCFD_PATH)
ensure_directory_exists(TEMP_PATH)

paper_case_number = None

paper_content = " "
paper_table = " "

boundary_type_match = None

global_OF_keywords = None

best_reference_cases = []

of_tutorial_dir = "/usr/lib/openfoam/openfoam2406/tutorials"  # '/home/fane/OpenFOAM/fane-v2406/tutorials' 

global_file_requirement = {}

global_files = None

pdf_short_case_description = None

OF_data_path = f"{Database_OFv24_PATH}/processed_merged_OF_cases.json"

OF_case_data_dict = {}      # Content from f"{Database_OFv24_PATH}/processed_merged_OF_cases.json"

max_running_test_round = 30

general_prompts = '''Respond to the following user query in a comprehensive and detailed way. You can write down your thought process before responding. Write your thoughts after “Here is my thought process:” and write your response after “Here is my response:”. \n'''

steady_solvers = ["laplacianFoam", "overLaplacianDyMFoam","potentialFoam","overPotentialFoam","scalarTransportFoam","adjointShapeOptimizationFoam","boundaryFoam","simpleFoam","overSimpleFoam","porousSimpleFoam","SRFSimpleFoam","rhoSimpleFoam","overRhoSimpleFoam","rhoPorousSimpleFoam","interFoam","interMixingFoam","interIsoFoam","interPhaseChangeFoam","MPPICInterFoam","multiphaseInterFoam","potentialFreeSurfaceFoam","potentialFreeSurfaceDyMFoam","buoyantBoussinesqSimpleFoam","buoyantFoam", "buoyantSimpleFoam","chtMultiRegionSimpleFoam","thermoFoam","icoUncoupledKinematicParcelFoam","simpleReactingParcelFoam","simpleCoalParcelFoam","simpleSprayFoam","uncoupledKinematicParcelFoam", "solidEquilibriumDisplacementFoam","financialFoam"]

solver_keywords = ['buoyantBoussinesqSimpleFoam', 'overInterDyMFoam', 'kinematicParcelFoam', 
                   'buoyantSimpleFoam', 'reactingFoam', 'rhoReactingFoam', 
                   'compressibleInterDyMFoam', 'XiEngineFoam', 'pimpleFoam', 
                   'cavitatingFoam', 'adjointOptimisationFoam', 'overPimpleDyMFoam', 
                   'twoPhaseEulerFoam', 'interMixingFoam', 'compressibleInterFoam', 
                   'multiphaseInterFoam', 'porousSimpleFoam', 'overRhoSimpleFoam', 
                   'overLaplacianDyMFoam', 'interFoam', 'MPPICInterFoam', 'icoFoam', 
                   'overRhoPimpleDyMFoam', 'interPhaseChangeDyMFoam', 'sonicDyMFoam', 
                   'chtMultiRegionTwoPhaseEulerFoam', 'adjointShapeOptimizationFoam', 
                   'laplacianFoam', 'fireFoam', 'rhoSimpleFoam', 'overCompressibleInterDyMFoam', 
                   'shallowWaterFoam', 'simpleFoam', 'snappyHexMesh', 'sonicLiquidFoam', 
                   'sonicFoam', 'icoUncoupledKinematicParcelFoam', 'overSimpleFoam', 
                   'driftFluxFoam', 'interIsoFoam', 'uncoupledKinematicParcelDyMFoam', 
                   'sprayFoam', 'buoyantPimpleFoam', 'reactingHeterogenousParcelFoam', 
                   'chemFoam', 'acousticFoam', 'nonNewtonianIcoFoam', 'simpleReactingParcelFoam', 
                   'overInterPhaseChangeDyMFoam', 'boundaryFoam', 'compressibleMultiphaseInterFoam', 
                   'coalChemistryFoam', 'coldEngineFoam', 'rhoPimpleAdiabaticFoam', 
                   'MPPICDyMFoam', 'icoReactingMultiphaseInterFoam', 'SRFPimpleFoam', 
                   'overBuoyantPimpleDyMFoam', 'solidFoam', 'reactingParcelFoam', 
                   'icoUncoupledKinematicParcelDyMFoam', 'compressibleInterIsoFoam', 
                   'potentialFreeSurfaceFoam', 'chtMultiRegionFoam', 'XiDyMFoam', 
                   'multiphaseEulerFoam', 'overPotentialFoam', 'interCondensatingEvaporatingFoam', 
                   'potentialFreeSurfaceDyMFoam', 'subsetMesh', 'twoLiquidMixingFoam', 
                   'rhoPimpleFoam', 'MPPICFoam', 'pisoFoam', 'potentialFoam', 
                   'reactingTwoPhaseEulerFoam', 'reactingMultiphaseEulerFoam', 
                   'rhoPorousSimpleFoam', 'rhoCentralFoam', 'SRFSimpleFoam', 
                   'PDRFoam', 'interPhaseChangeFoam', 'buoyantBoussinesqPimpleFoam', 
                   'XiFoam', 'dnsFoam', 'chtMultiRegionSimpleFoam', 'buoyantFoam']

turbulence_type_keywords = ['laminar', 'RAS', 'LES', 'twoPhaseTransport']

turbulence_model_keywords = ['SpalartAllmarasDDES', 'Smagorinsky', 'SpalartAllmaras', 
                                        'SpalartAllmarasIDDES', 'kOmegaSST', 'buoyantKEpsilon', 
                                        'kkLOmega', 'RNGkEpsilon', 'WALE', 'LaunderSharmaKE', 
                                        'realizableKE', 'PDRkEpsilon', 'dynamicKEqn', 
                                        'kOmegaSSTLM', 'kEqn', 'kEpsilon', 'kEpsilonPhitF']

boundary_type_keywords = ['overset', 'zeroGradient', 'fixedValue', 'movingWallVelocity', 'inletOutlet', 'symmetryPlane', 'symmetry', 'empty', 'uniformFixedValue', 'noSlip', 'cyclicAMI', 'mappedField', 'calculated', 'waveTransmissive', 'compressible::alphatWallFunction', 'supersonicFreestream', 'epsilonWallFunction', 'kqRWallFunction', 'nutkWallFunction', 'slip', 'turbulentIntensityKineticEnergyInlet', 'turbulentMixingLengthDissipationRateInlet', 'flowRateInletVelocity', 'freestreamPressure', 'omegaWallFunction', 'freestreamVelocity', 'pressureInletOutletVelocity', 'nutUWallFunction', 'totalPressure', 'wedge', 'totalTemperature', 'turbulentInlet', 'fixedMean', 'plenumPressure', 'pressureInletVelocity', 'fluxCorrectedVelocity', 'mixed', 'uniformTotalPressure', 'outletMappedUniformInletHeatAddition', 'clampedPlate', 'nutUSpaldingWallFunction', 'compressible::turbulentTemperatureTwoPhaseRadCoupledMixed', 'copiedFixedValue', 'prghTotalPressure', 'fixedFluxPressure', 'lumpedMassWallTemperature', 'greyDiffusiveRadiation', 'compressible::turbulentTemperatureRadCoupledMixed', 'externalWallHeatFluxTemperature', 'fixedGradient', 'humidityTemperatureCoupledMixed', 'wideBandDiffusiveRadiation', 'greyDiffusiveRadiationViewFactor', 'alphatJayatillekeWallFunction', 'processor', 'compressible::thermalBaffle', 'compressible::alphatJayatillekeWallFunction', 'prghPressure', 'MarshakRadiation', 'surfaceNormalFixedValue', 'turbulentMixingLengthFrequencyInlet', 'interstitialInletVelocity', 'JohnsonJacksonParticleSlip', 'JohnsonJacksonParticleTheta', 'mapped', 'fixedMultiPhaseHeatFlux', 'alphaContactAngle', 'permeableAlphaPressureInletOutletVelocity', 'prghPermeableAlphaTotalPressure', 'nutkRoughWallFunction', 'constantAlphaContactAngle', 'waveAlpha', 'waveVelocity', 'variableHeightFlowRate', 'outletPhaseMeanVelocity', 'variableHeightFlowRateInletVelocity', 'rotatingWallVelocity', 'cyclic', 'porousBafflePressure', 'translatingWallVelocity', 'multiphaseEuler::alphaContactAngle', 'pressureInletOutletParSlipVelocity', 'waveSurfacePressure', 'flowRateOutletVelocity', 'timeVaryingMassSorption', 'adjointOutletPressure', 'adjointOutletVelocity', 'SRFVelocity', 'adjointFarFieldPressure', 'adjointInletVelocity', 'adjointWallVelocity', 'adjointInletNuaTilda', 'adjointOutletNuaTilda', 'nutLowReWallFunction', 'outletInlet', 'freestream', 'adjointFarFieldVelocity', 'adjointFarFieldNuaTilda', 'waWallFunction', 'adjointZeroInlet', 'adjointOutletWa', 'kaqRWallFunction', 'adjointOutletKa', 'adjointFarFieldTMVar2', 'adjointFarFieldTMVar1', 'adjointOutletVelocityFlux', 'adjointOutletNuaTildaFlux', 'SRFFreestreamVelocity', 'timeVaryingMappedFixedValue', 'atmBoundaryLayerInletVelocity', 'atmBoundaryLayerInletEpsilon', 'atmBoundaryLayerInletK', 'atmNutkWallFunction', 'nutUBlendedWallFunction', 'maxwellSlipU', 'smoluchowskiJumpT', 'freeSurfacePressure', 'freeSurfaceVelocity']

boundary_required_field = [{'type': 'overset', 'require_entry': ['value']}, {'type': 'zeroGradient', 'require_entry': ['value']}, {'type': 'fixedValue', 'require_entry': []}, {'type': 'movingWallVelocity', 'require_entry': []}, {'type': 'inletOutlet', 'require_entry': []}, {'type': 'symmetryPlane', 'require_entry': ['value']}, {'type': 'symmetry', 'require_entry': []}, {'type': 'empty', 'require_entry': []}, {'type': 'uniformFixedValue', 'require_entry': []}, {'type': 'noSlip', 'require_entry': []}, {'type': 'cyclicAMI', 'require_entry': []}, {'type': 'mappedField', 'require_entry': []}, {'type': 'calculated', 'require_entry': []}, {'type': 'waveTransmissive', 'require_entry': []}, {'type': 'compressible::alphatWallFunction', 'require_entry': ['Prt']}, {'type': 'supersonicFreestream', 'require_entry': []}, {'type': 'epsilonWallFunction', 'require_entry': []}, {'type': 'kqRWallFunction', 'require_entry': []}, {'type': 'nutkWallFunction', 'require_entry': []}, {'type': 'slip', 'require_entry': ['value']}, {'type': 'turbulentIntensityKineticEnergyInlet', 'require_entry': []}, {'type': 'turbulentMixingLengthDissipationRateInlet', 'require_entry': []}, {'type': 'flowRateInletVelocity', 'require_entry': ['massFlowRate', 'value']}, {'type': 'freestreamPressure', 'require_entry': []}, {'type': 'omegaWallFunction', 'require_entry': []}, {'type': 'freestreamVelocity', 'require_entry': []}, {'type': 'pressureInletOutletVelocity', 'require_entry': []}, {'type': 'nutUWallFunction', 'require_entry': []}, {'type': 'totalPressure', 'require_entry': []}, {'type': 'wedge', 'require_entry': []}, {'type': 'totalTemperature', 'require_entry': []}, {'type': 'turbulentInlet', 'require_entry': []}, {'type': 'fixedMean', 'require_entry': []}, {'type': 'plenumPressure', 'require_entry': []}, {'type': 'pressureInletVelocity', 'require_entry': []}, {'type': 'fluxCorrectedVelocity', 'require_entry': []}, {'type': 'mixed', 'require_entry': []}, {'type': 'uniformTotalPressure', 'require_entry': []}, {'type': 'outletMappedUniformInletHeatAddition', 'require_entry': []}, {'type': 'clampedPlate', 'require_entry': []}, {'type': 'nutUSpaldingWallFunction', 'require_entry': []}, {'type': 'compressible::turbulentTemperatureTwoPhaseRadCoupledMixed', 'require_entry': []}, {'type': 'copiedFixedValue', 'require_entry': []}, {'type': 'prghTotalPressure', 'require_entry': []}, {'type': 'fixedFluxPressure', 'require_entry': []}, {'type': 'lumpedMassWallTemperature', 'require_entry': []}, {'type': 'greyDiffusiveRadiation', 'require_entry': []}, {'type': 'compressible::turbulentTemperatureRadCoupledMixed', 'require_entry': ['qr', 'qrNbr', 'kappa']}, {'type': 'externalWallHeatFluxTemperature', 'require_entry': ['q', 'kappaName']}, {'type': 'fixedGradient', 'require_entry': []}, {'type': 'humidityTemperatureCoupledMixed', 'require_entry': []}, {'type': 'wideBandDiffusiveRadiation', 'require_entry': []}, {'type': 'greyDiffusiveRadiationViewFactor', 'require_entry': []}, {'type': 'alphatJayatillekeWallFunction', 'require_entry': []}, {'type': 'processor', 'require_entry': ['value']}, {'type': 'compressible::thermalBaffle', 'require_entry': []}, {'type': 'compressible::alphatJayatillekeWallFunction', 'require_entry': []}, {'type': 'prghPressure', 'require_entry': []}, {'type': 'MarshakRadiation', 'require_entry': []}, {'type': 'surfaceNormalFixedValue', 'require_entry': ['value']}, {'type': 'turbulentMixingLengthFrequencyInlet', 'require_entry': ['k']}, {'type': 'interstitialInletVelocity', 'require_entry': []}, {'type': 'JohnsonJacksonParticleSlip', 'require_entry': []}, {'type': 'JohnsonJacksonParticleTheta', 'require_entry': []}, {'type': 'mapped', 'require_entry': []}, {'type': 'fixedMultiPhaseHeatFlux', 'require_entry': []}, {'type': 'alphaContactAngle', 'require_entry': []}, {'type': 'permeableAlphaPressureInletOutletVelocity', 'require_entry': []}, {'type': 'prghPermeableAlphaTotalPressure', 'require_entry': []}, {'type': 'nutkRoughWallFunction', 'require_entry': []}, {'type': 'constantAlphaContactAngle', 'require_entry': []}, {'type': 'waveAlpha', 'require_entry': []}, {'type': 'waveVelocity', 'require_entry': []}, {'type': 'variableHeightFlowRate', 'require_entry': []}, {'type': 'outletPhaseMeanVelocity', 'require_entry': []}, {'type': 'variableHeightFlowRateInletVelocity', 'require_entry': []}, {'type': 'rotatingWallVelocity', 'require_entry': []}, {'type': 'cyclic', 'require_entry': ['value']}, {'type': 'porousBafflePressure', 'require_entry': []}, {'type': 'translatingWallVelocity', 'require_entry': []}, {'type': 'multiphaseEuler::alphaContactAngle', 'require_entry': []}, {'type': 'pressureInletOutletParSlipVelocity', 'require_entry': []}, {'type': 'waveSurfacePressure', 'require_entry': []}, {'type': 'flowRateOutletVelocity', 'require_entry': []}, {'type': 'timeVaryingMassSorption', 'require_entry': []}, {'type': 'adjointOutletPressure', 'require_entry': []}, {'type': 'adjointOutletVelocity', 'require_entry': []}, {'type': 'SRFVelocity', 'require_entry': []}, {'type': 'adjointFarFieldPressure', 'require_entry': []}, {'type': 'adjointInletVelocity', 'require_entry': []}, {'type': 'adjointWallVelocity', 'require_entry': []}, {'type': 'adjointInletNuaTilda', 'require_entry': []}, {'type': 'adjointOutletNuaTilda', 'require_entry': []}, {'type': 'nutLowReWallFunction', 'require_entry': []}, {'type': 'outletInlet', 'require_entry': []}, {'type': 'freestream', 'require_entry': []}, {'type': 'adjointFarFieldVelocity', 'require_entry': []}, {'type': 'adjointFarFieldNuaTilda', 'require_entry': []}, {'type': 'waWallFunction', 'require_entry': []}, {'type': 'adjointZeroInlet', 'require_entry': []}, {'type': 'adjointOutletWa', 'require_entry': []}, {'type': 'kaqRWallFunction', 'require_entry': []}, {'type': 'adjointOutletKa', 'require_entry': []}, {'type': 'adjointFarFieldTMVar2', 'require_entry': []}, {'type': 'adjointFarFieldTMVar1', 'require_entry': []}, {'type': 'adjointOutletVelocityFlux', 'require_entry': []}, {'type': 'adjointOutletNuaTildaFlux', 'require_entry': []}, {'type': 'SRFFreestreamVelocity', 'require_entry': []}, {'type': 'timeVaryingMappedFixedValue', 'require_entry': []}, {'type': 'atmBoundaryLayerInletVelocity', 'require_entry': []}, {'type': 'atmBoundaryLayerInletEpsilon', 'require_entry': []}, {'type': 'atmBoundaryLayerInletK', 'require_entry': []}, {'type': 'atmNutkWallFunction', 'require_entry': []}, {'type': 'nutUBlendedWallFunction', 'require_entry': []}, {'type': 'maxwellSlipU', 'require_entry': []}, {'type': 'smoluchowskiJumpT', 'require_entry': []}, {'type': 'freeSurfacePressure', 'require_entry': []}, {'type': 'freeSurfaceVelocity', 'require_entry': []}]

thermodynamic_model_keywords = ['hePsiThermo', 'heRhoThermo', 'heSolidThermo']

string_of_turbulence_type_keywords= ", ".join(turbulence_type_keywords)
string_of_turbulence_model= ", ".join(turbulence_model_keywords)
string_of_solver_keywords = ", ".join(solver_keywords)
string_of_boundary_type_keywords = ", ".join(boundary_type_keywords)
string_of_thermodynamic_model = ", ".join(thermodynamic_model_keywords)

case_log_write = False

flag_OF_tutorial_processed = False

mesh_convert_success = False

set_controlDict_time = False

pdf_chunk_d = 1.5

case_ic_bc_from_paper = ""

reference_file_by_name = None
reference_file_by_solver = None

simulate_requirement = None     # Case running requirement settings (case_name, case_solver, turbulence_model, case_description)
boundary_name_and_type = None

incompressible_solvers = [
    # Basic incompressible flow solvers
    "simpleFoam",           # Steady-state incompressible flow
    "pimpleFoam",           # Transient incompressible flow
    "pisoFoam",             # Transient incompressible flow (PISO algorithm)
    "icoFoam",              # Isothermal incompressible flow
    "adjointOptimisationFoam",
    "adjointShapeOptimizationFoam",
    "boundaryFoam",
    "lumpedPointMotion",
    "nonNewtonianlcoFoam",
    "overPimpleDyMFoam",
    "overSimpleFoam",
    "porousSimpleFoam",
    "shallowWaterFoam",
    "SRFPimpleFoam",
    "SRFSimpleFoam",

    # Special applications
    "potentialFoam",        # Potential flow initialization
]

# Compressible Flow Solvers
compressible_solvers = [
    # Basic compressible flow solvers
    "rhoSimpleFoam",        # Steady-state compressible flow
    "rhoPimpleFoam",        # Transient compressible flow
    "rhoCentralFoam",       # Compressible flow based on central schemes
    "sonicFoam",            # Transient compressible flow (subsonic/supersonic)
    "sonicLiquidFoam",      # Compressible liquid flow
    "acousticFoam",
    "overRhoPimpleDyMFoam",
    "overRhoSimpleFoam",
    "overRhoSimpleFoam",
    "rhoPimpleAdiabaticFoam",
    "rhoPorousSimpleFoam",
    "sonicDyMFoam",
    "sonicLiquidFoam",

    # Combustion and chemical reaction
    "reactingFoam",         # Reactive flow
    
    # Multiphase compressible flow
    "interFoam",            # Multiphase flow using VOF method
    "compressibleInterFoam", # Compressible multiphase flow
    "twoPhaseCompressibleFoam", # Two-phase compressible flow
    "compressibleInterIsoFoam",
    "MPPICInterFoam",
    "overCompressibleInterDyMFoam"
        
]
