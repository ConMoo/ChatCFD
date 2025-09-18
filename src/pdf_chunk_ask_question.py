import pdfplumber
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from openai import OpenAI
from pdfplumber.utils import within_bbox
import re
import tiktoken
from datetime import datetime
import qa_modules, config, os


class CFDCaseExtractor:
    def __init__(self, model_name=config.sentence_transformer_path):
        if not os.path.exists(config.sentence_transformer_path):
            model_name='sentence-transformers/all-mpnet-base-v2'
        self.embedder = SentenceTransformer(model_name)
        self.client = OpenAI(
            api_key=os.environ.get("DEEPSEEK_R1_KEY"), 
            base_url=os.environ.get("DEEPSEEK_R1_BASE_URL")
        )
        self.gpt_model = os.environ.get("DEEPSEEK_R1_MODEL_NAME")
        self.index = None
        self.chunks = []
        self.token_usage = []  # Added Token usage statistics storage
        self.encoder = tiktoken.encoding_for_model("gpt-4")

    def process_pdf(self, file_path):
        """Optimized PDF processing workflow (fixed bbox errors)"""
        if file_path.endswith('.pdf'):
            with pdfplumber.open(file_path) as pdf:
                text_blocks = []
                for i, page in enumerate(pdf.pages):
                    # Initialize QA module based on context parameterDefine valid text area (unit: points)
                    bbox = (
                        50,  # left margin
                        50,  # top margin (skip header)
                        page.width - 50,  # right margin
                        page.height - 50  # bottom margin (skip footer)
                    )
                    
                    # Create filter function (key fix)
                    crop_filter = lambda obj: (
                        obj["x0"] >= bbox[0] and
                        obj["top"] >= bbox[1] and
                        obj["x1"] <= bbox[2] and
                        obj["bottom"] <= bbox[3]
                    )
                    
                    # Apply area filtering
                    cropped_page = page.filter(crop_filter)
                    
                    # Optimize text extraction parameters
                    text = cropped_page.extract_text(
                        layout=True,
                        x_tolerance=3,
                        y_tolerance=2,
                        keep_blank_chars=False,
                        extra_attrs=["size", "fontname"]
                    )
                    
                    # Text cleaning
                    cleaned_text = self.clean_text(text, page_number=i+1)
                    if cleaned_text:
                        text_blocks.append(cleaned_text)
            
            # Intelligent chunking strategy
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=600,
                chunk_overlap=100,
                separators=[
                    r"\n\s*[A-Z][A-Z\s]+\s*:\s*\n",  # Match titles like "METHODOLOGY:"
                    r"\n\s*\d+\.\s*[A-Z]",          # Match section numbers like "3. RESULTS"
                    "\n\n"
                ]
            )
            self.chunks = splitter.split_text("\n".join(text_blocks))
            
            # Filter empty blocks and short text
            self.chunks = [chunk for chunk in self.chunks 
                        if len(chunk.strip()) > 50]
        elif file_path.endswith('.txt'):
            with open(file_path, 'r', encoding='utf-8') as file:
                text = file.read()
                # Process text file content
                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=600,
                    chunk_overlap=100,
                    separators=["\n\n", "\n", " ", ""]
                )
                self.chunks = splitter.split_text(text.strip())
                self.chunks = [text.strip()]

        # Create FAISS index
        embeddings = self.embedder.encode(self.chunks, 
                                        convert_to_numpy=True,
                                        show_progress_bar=False)
        embeddings = np.array(embeddings).astype('float32')
        
        self.index = faiss.IndexFlatL2(embeddings.shape[1])
        self.index.add(embeddings)
        

    def clean_text(self, text, page_number):
        """Multi-stage text cleaning"""
        # Stage 1: Merge broken words
        text = re.sub(r'(?<=\w)-\n(?=\w)', '', text)  # Connect words split by line breaks
        
        # Stage 2: Handle numbers and units
        text = re.sub(r'\n(?=\d+\s*[A-Za-z]{1,3}\b)', ' ', text)  # Fix unit line breaks
        
        # Stage 3: Remove isolated page numbers
        text = re.sub(r'^\s*\d+\s*$', '', text, flags=re.MULTILINE)
        
        # Stage 4: Compress whitespace
        text = re.sub(r'\n{3,}', '\n\n', text)  # Compress multiple line breaks to two
        text = re.sub(r'[ \t]{2,}', ' ', text)   # Compress multiple spaces to one
        
        # Stage 5: Filter short text segments (possibly chart annotations)
        lines = [line.strip() for line in text.split('\n') if len(line.strip()) > 3]
        
        # Add page metadata
        return f"Page {page_number}:\n" + "\n".join(lines) if lines else ""

    def _count_tokens(self, text):
        """Use Tiktoken to accurately calculate tokens"""
        return len(self.encoder.encode(text))

    def query_case_setup(self, question, detailed_question = None, top_k=3, context = False):
        """Enhanced query method with Token statistics
        Args:
            question (str): User's question, used for embedding to find relevant chunks
            detailed_question (str): Detailed question for LLM to answer (usually includes question)
            top_k (int): Number of relevant chunks to return
            context (bool): Whether to use context
        Return:
            R1_response (str): LLM's response
        """
        if detailed_question == None:
            detailed_question = question

        try:
            # Initialize request record
            request_entry = {
                "timestamp": datetime.now().isoformat(),
                "question": question,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
                "status": "success",
                "error": None
            }

            if not self.index:
                raise ValueError("Please use process_pdf to process PDF documents first")

            # Semantic retrieval stage
            query_embed = self.embedder.encode([question])
            distances, indices = self.index.search(query_embed, top_k)
            relevant_chunks = [self.chunks[i] for i, d in zip(indices[0], distances[0]) if d < config.pdf_chunk_d]
            relevant_chunks_loose = [self.chunks[i] for i in indices[0]]
            if not relevant_chunks:
                relevant_chunks = relevant_chunks_loose

            # Calculate context token consumption
            context_tokens = sum(self._count_tokens(chunk) for chunk in relevant_chunks)
            request_entry["context_tokens"] = context_tokens

            prompt = f'''You are a CFD expert assistant. Extract technical parameters from research papers and structure answers in markdown tables.
            Analyze these CFD paper excerpts:
            [[[ {relevant_chunks} ]]]
            Extract specific details about: [[[ {detailed_question} ]]] 
            '''

            qa = None

            if context == True:
                qa = qa_modules.QA_Context_deepseek_R1()
            else:
                qa = qa_modules.QA_NoContext_deepseek_R1()

            R1_response = qa.ask(prompt)

            return R1_response

        except Exception as e:
            request_entry.update({
                "status": "failed",
                "error": str(e)
            })
            self.token_usage.append(request_entry)
            return f"Processing exception: {str(e)}"
