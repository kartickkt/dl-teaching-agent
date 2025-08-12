import os
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

MODEL_NAME = os.environ.get("MODEL_NAME", "mistralai/Mistral-7B-Instruct-v0.3")
LOCAL_FALLBACK = "gpt2"

def get_pipeline():
    model_name = MODEL_NAME
    try:
        # Mistral-7B: recommended to run on Colab GPU with bitsandbytes installed
        if "mistralai" in model_name.lower():
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                load_in_4bit=True,
                device_map="auto"
            )
        else:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForCausalLM.from_pretrained(model_name)
        return pipeline("text-generation", model=model, tokenizer=tokenizer)
    except Exception as e:
        print("Model load failed:", e)
        print("Falling back to small CPU model for local testing.")
        tokenizer = AutoTokenizer.from_pretrained(LOCAL_FALLBACK)
        model = AutoModelForCausalLM.from_pretrained(LOCAL_FALLBACK)
        return pipeline("text-generation", model=model, tokenizer=tokenizer)

if __name__ == "__main__":
    pipe = get_pipeline()
    print(pipe("Explain Newton's second law to a 16-year-old.", max_length=150)[0]['generated_text'])
