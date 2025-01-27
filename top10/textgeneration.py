# -*- coding: utf-8 -*-
""" 
EnergyPlus IDF Generation with Open-Source Models (Colab-Compatible)
"""
# Install all required libraries first
!pip install -q transformers accelerate bitsandbytes torch

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    pipeline
)

# --------------------------
# 1. Mistral-7B-Instruct (4-bit Quantized)
# --------------------------
def mistral_7b_idf_generation(prompt):
    model_name = "mistralai/Mistral-7B-Instruct-v0.1"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        device_map="auto", 
        load_in_4bit=True
    )
    
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)
    output = pipe(prompt, max_new_tokens=200)
    return output[0]['generated_text']

# Example usage:
# print(mistral_7b_idf_generation("Generate an IDF SimulationControl object with zone sizing enabled."))

# --------------------------
# 2. Microsoft Phi-3-mini (3.8B)
# --------------------------
def phi3_mini_idf_generation(prompt):
    model = AutoModelForCausalLM.from_pretrained(
        "microsoft/Phi-3-mini-4k-instruct", 
        trust_remote_code=True,
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-4k-instruct")
    
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    outputs = model.generate(**inputs, max_length=200)
    return tokenizer.decode(outputs[0])

# Example usage:
# print(phi3_mini_idf_generation("Create a Site:Location object for Berlin."))

# --------------------------
# 3. Google Gemma-2B
# --------------------------
def gemma_2b_idf_generation(prompt):
    model = AutoModelForCausalLM.from_pretrained(
        "google/gemma-2b-it",
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained("google/gemma-2b-it")
    
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    outputs = model.generate(**inputs, max_new_tokens=200)
    return tokenizer.decode(outputs[0])

# Example usage:
# print(gemma_2b_idf_generation("Generate a ShadowCalculation object with PixelCounting method."))

# --------------------------
# 4. Zephyr-7B (4-bit Quantized)
# --------------------------
def zephyr_7b_idf_generation(prompt):
    pipe = pipeline(
        "text-generation",
        model="HuggingFaceH4/zephyr-7b-beta",
        device_map="auto",
        model_kwargs={"load_in_4bit": True}
    )
    output = pipe(prompt, max_new_tokens=200)
    return output[0]['generated_text']

# Example usage:
# print(zephyr_7b_idf_generation("Write an IDF SizingPeriod:DesignDay for a summer day in Tokyo."))

# --------------------------
# 5. TinyLlama-1.1B
# --------------------------
def tinyllama_idf_generation(prompt):
    model = AutoModelForCausalLM.from_pretrained(
        "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    outputs = model.generate(**inputs, max_length=200)
    return tokenizer.decode(outputs[0])

# Example usage:
# print(tinyllama_idf_generation("Generate a SimulationControl object for EnergyPlus."))

# --------------------------
# 6. StableLM-3B
# --------------------------
def stablelm_3b_idf_generation(prompt):
    model = AutoModelForCausalLM.from_pretrained(
        "stabilityai/stablelm-3b-4e1t",
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained("stabilityai/stablelm-3b-4e1t")
    
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    outputs = model.generate(**inputs, max_length=200)
    return tokenizer.decode(outputs[0])

# Example usage:
# print(stablelm_3b_idf_generation("Create an IDF Site:Location object for London."))

# --------------------------
# 7. Falcon-7B (4-bit)
# --------------------------
def falcon_7b_idf_generation(prompt):
    pipe = pipeline(
        "text-generation",
        model="tiiuae/falcon-7b-instruct",
        device_map="auto",
        model_kwargs={"load_in_4bit": True}
    )
    output = pipe(prompt, max_new_tokens=200)[0]['generated_text']
    return output

# Example usage:
# print(falcon_7b_idf_generation("Generate a ShadowCalculation object with SutherlandHodgman clipping."))

# --------------------------
# 8. Llama-2-7B (4-bit) - REQUIRES HUGGING FACE LOGIN
# --------------------------
# !pip install -q huggingface_hub
# from huggingface_hub import notebook_login
# notebook_login()

# def llama2_7b_idf_generation(prompt):
#     model = "meta-llama/Llama-2-7b-chat-hf"
#     tokenizer = AutoTokenizer.from_pretrained(model)
#     model = AutoModelForCausalLM.from_pretrained(
#         model, 
#         device_map="auto", 
#         load_in_4bit=True
#     )
    
#     inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
#     outputs = model.generate(**inputs, max_new_tokens=200)
#     return tokenizer.decode(outputs[0])

# Example usage:
# print(llama2_7b_idf_generation("Generate an IDF DesignDay with max temp 35°C."))

# --------------------------
# 9. GPT-NeoXT-1.3B
# --------------------------
def gpt_neox_idf_generation(prompt):
    generator = pipeline(
        "text-generation", 
        model="EleutherAI/gpt-neoxt-1.3B",
        device_map="auto"
    )
    output = generator(prompt, max_length=200)
    return output[0]['generated_text']

# Example usage:
# print(gpt_neox_idf_generation("Create a Site:Location object for Paris."))

# --------------------------
# 10. Pythia-2.8B
# --------------------------
def pythia_28b_idf_generation(prompt):
    model = AutoModelForCausalLM.from_pretrained(
        "EleutherAI/pythia-2.8b",
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-2.8b")
    
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    outputs = model.generate(**inputs, max_length=200)
    return tokenizer.decode(outputs[0])

# Example usage:
# print(pythia_28b_idf_generation("Generate an IDF SimulationControl object."))

# ======================
# MAIN EXECUTION
# ======================
if __name__ == "__main__":
    # Test one model at a time (Colab free tier has limited resources)
    prompt = "Generate an IDF Site:Location for New York with latitude 40.71, longitude -74.01"
    
    # Uncomment one of these to test:
    # print("Mistral-7B:", mistral_7b_idf_generation(prompt))
    # print("Phi-3:", phi3_mini_idf_generation(prompt))
    # print("Gemma-2B:", gemma_2b_idf_generation(prompt))
  
