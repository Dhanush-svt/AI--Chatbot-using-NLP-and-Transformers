from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

class AIChatbot:
    def __init__(self):
        self.model_name = "microsoft/DialoGPT-medium"
        self.max_history_turns = 6
        self.conversation_history = []
        print("Loading model...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = self.model.to(self.device)
        print(f"Ready! Running on: {self.device}")

    def generate_response(self, user_input):
        self.conversation_history.append(
            user_input + self.tokenizer.eos_token
        )
        recent = self.conversation_history[-self.max_history_turns:]
        context = "".join(recent)
        input_ids = self.tokenizer.encode(
            context,
            return_tensors="pt",
            truncation=True,
            max_length=900
        ).to(self.device)
        with torch.no_grad():
            output_ids = self.model.generate(
                input_ids,
                max_new_tokens=150,
                do_sample=True,
                temperature=0.75,
                top_k=50,
                top_p=0.92,
                repetition_penalty=1.3,
                pad_token_id=self.tokenizer.eos_token_id
            )
        response = self.tokenizer.decode(
            output_ids[:, input_ids.shape[-1]:][0],
            skip_special_tokens=True
        )
        self.conversation_history.append(
            response + self.tokenizer.eos_token
        )
        return response

    def reset_conversation(self):
        self.conversation_history = []
        print("Conversation cleared.")
