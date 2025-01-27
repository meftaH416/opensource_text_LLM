Here are the best open-source text-generation models that can run on Google Colab’s free tier (CPU/T4 GPU) for generating EnergyPlus IDF files, ordered by performance and ease of use: Suggeated by DeepSeek

Tips for Colab Free Tier:
Quantization: Use 4-bit (load_in_4bit=True) to reduce VRAM usage.
Batch Size: Set batch_size=1 to avoid OOM errors.
Model Swapping: Use device_map="auto" to let Hugging Face manage resources.
Max Tokens: Keep max_new_tokens ≤ 200 for IDF snippets.

Best for text Generation:
Mistral-7B-4bit (best quality)
Phi-3-mini (best speed/accuracy tradeoff)
TinyLlama (fastest on low RAM)

Most of the open-source models listed earlier can be fine-tuned for EnergyPlus IDF generation. Below is a step-by-step guide with code snippets to fine-tune these models on your IDF dataset in Google Colab:

Models That Can Be Fine-Tuned:
Model	         Fine-Tunable?	     Best Method	    Colab-Friendly?
Mistral-7B	        ✅	            LoRA/QLoRA	       Yes (T4 GPU)
Phi-3-mini	        ✅	            Full Fine-Tuning	 Yes
Gemma-2B	        ✅	            LoRA	             Yes
TinyLlama-1.1B	        ✅	            Full Fine-Tuning	 Yes
Llama-2-7B	        ✅	            QLoRA	             Yes (T4 GPU)
StableLM-3B	        ✅	            LoRA	             Yes

Dataset:
{
  "input": "Create a SimulationControl object with zone sizing",
  "output": "SimulationControl,\n  DoZoneSizingCalculation, Yes;..."
}

Expected VRAM Usage for Phi-3 mini with QLoRa:
Component	         VRAM Usage
Base Model (4-bit)	  ~3.5 GB
LoRA Adapters	          ~0.5 GB
Training Overhead	  ~1 GB
Total	                  ~5 GB
