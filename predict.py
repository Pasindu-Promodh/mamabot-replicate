from cog import BasePredictor, Input
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import os

class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory - model should already be cached from build"""
        print("Loading model from cache...")
        
        # Model should already be downloaded during build
        # We don't need the token here since files are cached
        self.tokenizer = AutoTokenizer.from_pretrained(
            'HelpMumHQ/MamaBot-Llama',
            local_files_only=False  # Will use cached files if available
        )
        
        self.model = AutoModelForCausalLM.from_pretrained(
            'HelpMumHQ/MamaBot-Llama',
            torch_dtype=torch.bfloat16,
            device_map="auto",
            local_files_only=False  # Will use cached files if available
        )
        
        self.tokenizer.chat_template = "{%- for message in messages %}{{ bos_token + '[INST] ' + message['content'] + ' [/INST]' if message['role'] == 'user' else ' ' + message['content'] + ' ' + eos_token }}{%- endfor %}"
        
        print("Model loaded successfully!")

    def predict(
        self,
        prompt: str = Input(
            description="Your maternal healthcare question",
            default="Why might mothers not realize they are already pregnant in the first two weeks?"
        ),
        max_length: int = Input(
            description="Maximum length of generated response",
            default=150,
            ge=50,
            le=500
        ),
        temperature: float = Input(
            description="Temperature for sampling",
            default=0.7,
            ge=0.1,
            le=2.0
        )
    ) -> str:
        """Generate response to maternal healthcare question"""
        
        messages = [{"role": "user", "content": prompt}]
        
        formatted_prompt = self.tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
        
        inputs = self.tokenizer(
            formatted_prompt, 
            return_tensors='pt', 
            truncation=True
        ).to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=max_length,
                num_return_sequences=1,
                temperature=temperature,
                do_sample=True
            )
        
        text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        if '[/INST]' in text:
            response = text.split('[/INST]')[-1].strip()
            if '[INST]' in response:
                response = response.split('[INST]')[0].strip()
        else:
            response = text
            
        return response