from openai import OpenAI
import os
import config
from datetime import datetime
import tiktoken
import json

def estimate_tokens(text: str, model_name: str) -> int:
    """Estimate token count using tiktoken"""
    try:
        encoding = tiktoken.encoding_for_model(model_name)
    except KeyError:
        # If model is not recognized, default to cl100k_base (GPT-4 encoding)
        encoding = tiktoken.get_encoding("cl100k_base")
    return len(encoding.encode(text))

class GlobalLogManager:
    _instance = None
    current_session_stats = {
        "deepseek-v3": {"calls": 0, "prompt_tokens": 0, "response_tokens": 0},
        "deepseek-r1": {"calls": 0, "prompt_tokens": 0, "response_tokens": 0, "reasoning_tokens": 0}
    }
    
    @classmethod
    def add_log(cls, log_entry):
        # Only update statistics, do not save complete logs in memory
        model_type = log_entry["model_type"]
        if model_type in cls.current_session_stats:
            stats = cls.current_session_stats[model_type]
            stats["calls"] += 1
            stats["prompt_tokens"] += log_entry.get("prompt_tokens", 0)
            stats["response_tokens"] += log_entry.get("response_tokens", 0)
            if model_type == "deepseek-r1":
                stats["reasoning_tokens"] += log_entry.get("reasoning_tokens", 0)
        
        # Write directly to file, do not save in memory
        if config.case_log_write:
            cls._append_log_to_file(log_entry)
    
    @classmethod
    def _append_log_to_file(cls, log_entry):
        """Append log to file, avoid memory accumulation"""
        config.ensure_directory_exists(config.OUTPUT_PATH)
        log_file_path = f'{config.OUTPUT_PATH}/qa_logs.jsonl'  # Use JSONL format
        
        with open(log_file_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(log_entry, ensure_ascii=False, indent=2) + '\n')
    
    @classmethod
    def get_session_stats(cls):
        """Get current session statistics"""
        return cls.current_session_stats.copy()
    
    @classmethod
    def reset_session(cls):
        """Reset session statistics"""
        for model_stats in cls.current_session_stats.values():
            for key in model_stats:
                model_stats[key] = 0

class BaseQA_deepseek_V3:
    def __init__(self):
        self.qa_interface = self._setup_qa_interface()
        self._initialized = True

    def _setup_qa_interface(self):
        def get_deepseekV3_response(messages):
            client = OpenAI(
                api_key=os.environ.get("DEEPSEEK_V3_KEY"), 
                base_url=os.environ.get("DEEPSEEK_V3_BASE_URL")
            )

            chat_completion = client.chat.completions.create(
                messages=messages,
                model=os.environ.get("DEEPSEEK_V3_MODEL_NAME"),
                temperature=config.V3_temperature,
                stream=False
            )
            
            return {
                "content": chat_completion.choices[0].message.content,
                "prompt_tokens": chat_completion.usage.prompt_tokens,
                "completion_tokens": chat_completion.usage.completion_tokens
            }

        return get_deepseekV3_response

    def ask(self, question: str):
        raise NotImplementedError

    def close(self):
        pass

class QA_Context_deepseek_V3(BaseQA_deepseek_V3):
    def __init__(self):
        super().__init__()
        self.conversation_history: list[dict[str, str]] = []

    def ask(self, question: str):
        self.conversation_history.append({"role": "user", "content": question})
        result = self.qa_interface(self.conversation_history.copy())
        
        self.conversation_history.append({"role": "assistant", "content": result["content"]})
        
        GlobalLogManager.add_log({
            "model_type": "deepseek-v3",
            "user_prompt": question,
            "assistant_response": result["content"],
            "prompt_tokens": result["prompt_tokens"],
            "response_tokens": result["completion_tokens"],
            "timestamp": datetime.now().isoformat()
        })
        
        return result["content"]

class QA_NoContext_deepseek_V3(BaseQA_deepseek_V3):
    def ask(self, question: str):
        messages = [{"role": "user", "content": question}]
        result = self.qa_interface(messages)
        
        GlobalLogManager.add_log({
            "model_type": "deepseek-v3",
            "user_prompt": question,
            "assistant_response": result["content"],
            "prompt_tokens": result["prompt_tokens"],
            "response_tokens": result["completion_tokens"],
            "timestamp": datetime.now().isoformat()
        })
        
        return result["content"]

class BaseQA_deepseek_R1:
    def __init__(self):
        self.qa_interface = self._setup_qa_interface()
        self._initialized = True
        self.encoding = tiktoken.get_encoding("cl100k_base")

    def _setup_qa_interface(self):

        def get_response(messages):
            client = OpenAI(
                api_key=os.environ.get("DEEPSEEK_V3_KEY"),
                base_url=os.environ.get("DEEPSEEK_V3_BASE_URL")
            )

            # Get model name for token estimation
            model_name = os.environ.get("DEEPSEEK_R1_MODEL_NAME")
            
            # ===== Stream request to get content =====
            stream = client.chat.completions.create(
                messages=messages,
                model=model_name,
                temperature=config.R1_temperature,
                stream=True
            )

            full_content = []
            reasoning_contents = []
            
            for chunk in stream:
                if chunk.choices:
                    delta = chunk.choices[0].delta
                    if delta.content:
                        full_content.append(delta.content)
                    if hasattr(delta, 'model_extra') and 'reasoning_content' in delta.model_extra:
                        reasoning_contents.append(str(delta.model_extra['reasoning_content']))

            # ===== Estimate token usage =====
            # Estimate prompt tokens (serialize messages to string)
            prompt_str = json.dumps(messages, ensure_ascii=False)
            prompt_tokens = estimate_tokens(prompt_str, model_name)
            
            # Estimate completion tokens (actual returned content)
            completion_str = "".join(full_content)
            completion_tokens = estimate_tokens(completion_str, model_name)

            return {
                "reasoning_content": "".join(reasoning_contents),
                "answer": completion_str,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens
            }


        return get_response

    def ask(self, question: str):
        raise NotImplementedError

    def close(self):
        pass

class QA_Context_deepseek_R1(BaseQA_deepseek_R1):
    def __init__(self, system_prompt=None):
        super().__init__()
        if system_prompt:
            self.conversation_history = [{"role": "system", "content": system_prompt}]
        else:
            self.conversation_history: list[dict[str, str]] = []

    def ask(self, question: str):
        self.conversation_history.append({"role": "user", "content": question})
        result = self.qa_interface(self.conversation_history.copy())
        
        self.conversation_history.append({"role": "assistant", "content": result["answer"]})
        
        reasoning_tokens = len(self.encoding.encode(result["reasoning_content"]))
        
        GlobalLogManager.add_log({
            "model_type": "deepseek-r1",
            "user_prompt": question,
            "assistant_response": result["answer"],
            "reasoning_content": result["reasoning_content"],
            "prompt_tokens": result["prompt_tokens"],
            "response_tokens": result["completion_tokens"],
            "reasoning_tokens": reasoning_tokens,
            "timestamp": datetime.now().isoformat()
        })
        
        return result["answer"]

class QA_NoContext_deepseek_R1(BaseQA_deepseek_R1):
    def ask(self, question: str):
        messages = [{"role": "user", "content": question}]
        result = self.qa_interface(messages)
        
        reasoning_tokens = len(self.encoding.encode(result["reasoning_content"]))
        
        GlobalLogManager.add_log({
            "model_type": "deepseek-r1",
            "user_prompt": question,
            "assistant_response": result["answer"],
            "reasoning_content": result["reasoning_content"],
            "prompt_tokens": result["prompt_tokens"],
            "response_tokens": result["completion_tokens"],
            "reasoning_tokens": reasoning_tokens,
            "timestamp": datetime.now().isoformat()
        })
        
        return result["answer"]
