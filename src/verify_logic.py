"""
🧪 Neuro Brain — Verification Script for Cyber Reasoner
Validates the fine-tuned GPT-2 model's generation capabilities.
"""
import torch
from src.model_manager import ModelManager

def test_inference(prompt):
    manager = ModelManager()
    print(f"\n📂 Loading model for inference...")
    model, tokenizer = manager.load_cyber_reasoner()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    formatted_prompt = f"USER: {prompt}\nAI:"
    inputs = tokenizer(formatted_prompt, return_tensors="pt").to(device)
    
    print(f"🎯 Generating response for: \"{prompt}\"")
    with torch.no_grad():
        output_tokens = model.generate(
            **inputs, 
            max_new_tokens=100, 
            do_sample=True, 
            temperature=0.7,
            pad_token_id=tokenizer.eos_token_id
        )
    
    response = tokenizer.decode(output_tokens[0], skip_special_tokens=True)
    print("\n" + "="*60)
    print(response)
    print("="*60 + "\n")

if __name__ == "__main__":
    test_prompts = [
        "How do I validate an IP address in Python?",
        "Explain the importance of IP tracking in SOC."
    ]
    
    for prompt in test_prompts:
        test_inference(prompt)
