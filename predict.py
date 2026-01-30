# #predict.py
# from cog import BasePredictor, Input
# from transformers import AutoModelForCausalLM, AutoTokenizer
# import torch
# import os
# import accelerate


# class Predictor(BasePredictor):
#     def setup(self):
#         """Load the model into memory"""
#         print("Loading model...")
        
#         # Get HF token from environment (passed during build)
#         hf_token = os.environ.get("HUGGING_FACE_HUB_TOKEN")
        
#         # Load tokenizer
#         print("Downloading/loading tokenizer...")
#         self.tokenizer = AutoTokenizer.from_pretrained(
#             'HelpMumHQ/MamaBot-Llama',
#             token=hf_token
#         )
        
#         # Load model
#         print("Downloading/loading model (this may take a few minutes on first run)...")
#         self.model = AutoModelForCausalLM.from_pretrained(
#             'HelpMumHQ/MamaBot-Llama',
#             token=hf_token,
#             # torch_dtype=torch.bfloat16,
#             dtype=torch.bfloat16,
#             device_map="auto"
#         )
        
#         # Set chat template
#         self.tokenizer.chat_template = "{%- for message in messages %}{{ bos_token + '[INST] ' + message['content'] + ' [/INST]' if message['role'] == 'user' else ' ' + message['content'] + ' ' + eos_token }}{%- endfor %}"
        
#         print("Model loaded successfully!")

#     def predict(
#         self,
#         prompt: str = Input(
#             description="Your maternal healthcare question",
#             default="Why might mothers not realize they are already pregnant in the first two weeks?"
#         ),
#         max_length: int = Input(
#             description="Maximum length of generated response",
#             default=150,
#             ge=50,
#             le=500
#         ),
#         temperature: float = Input(
#             description="Temperature for sampling",
#             default=0.7,
#             ge=0.1,
#             le=2.0
#         )
#     ) -> str:
#         """Generate response to maternal healthcare question"""
        
#         messages = [{"role": "user", "content": prompt}]
        
#         formatted_prompt = self.tokenizer.apply_chat_template(
#             messages, 
#             tokenize=False, 
#             add_generation_prompt=True
#         )
        
#         inputs = self.tokenizer(
#             formatted_prompt, 
#             return_tensors='pt', 
#             truncation=True
#         ).to(self.model.device)
        
#         with torch.no_grad():
#             outputs = self.model.generate(
#                 **inputs,
#                 max_length=max_length,
#                 num_return_sequences=1,
#                 temperature=temperature,
#                 do_sample=True
#             )
        
#         text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
#         if '[/INST]' in text:
#             response = text.split('[/INST]')[-1].strip()
#             if '[INST]' in response:
#                 response = response.split('[INST]')[0].strip()
#         else:
#             response = text
            
#         return response






from cog import BasePredictor, Input
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import accelerate
import os

class Predictor(BasePredictor):
    def setup(self):
        hf_token = os.environ.get("HUGGING_FACE_HUB_TOKEN")

        self.tokenizer = AutoTokenizer.from_pretrained(
            "HelpMumHQ/MamaBot-Llama",
            token=hf_token
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            "HelpMumHQ/MamaBot-Llama",
            token=hf_token,
            dtype=torch.bfloat16,
            device_map="auto"
        )

    def predict(
        self,
        prompt: str = Input(default="Why might mothers not realize they are pregnant early?"),
        max_length: int = Input(default=150),
        temperature: float = Input(default=0.7),
    ) -> str:

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        outputs = self.model.generate(
            **inputs,
            max_length=max_length,
            temperature=temperature,
            do_sample=True,
        )

        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
