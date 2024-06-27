# Meta-Llama3-Intel-IPEX-LLM

## Overview
This Generative AI application was developed using the Intel Tiber Developer Cloud, fully utilizing IDC Jupyter Notebooks. The project integrates the Meta Llama 3 model and the Intel® Extension for PyTorch (IPEX) for model optimization.

## Setup Instructions

### Hugging Face Set Up
1. Login/Sign up to Hugging Face:
   - [Hugging Face](https://huggingface.co/)
2. Search **"meta-llama/Meta-Llama-3-8B-Instruct"**.
3. Review the conditions and agree to access the model (and wait for a few minutes).

### Generating Access Token
4. Click your profile > Settings > Access Tokens > Click on New Token.
5. Name your token (i.e: Meta Llama 3 Intel IPEX).
6. For the type, select "Read" from the dropdown menu.
7. Once generated, copy and save this Access Token. We will need it later.

### Environment Set Up
1. Sign-in/Create an **Intel® Tiber™ Developer Cloud** account:
   - [Intel Tiber Developer Cloud](http://cloud.intel.com/)
2. Inside the console, head over to the left menu bar and select the "Training" tab under the "Software" section.
3. On the top right corner, click on "Launch JupyterLab".

## Development

### Setups

1. **Create a new Conda environment:**
   ```
   conda create -n llm python=3.11 -y
2. **Activate the Conda environment:**
   ```
   conda activate llm
3. **Install Intel IPEX LLM and other dependencies:**
   ```
   pip install --pre --upgrade ipex-llm[all] --extra-index-url https://download.pytorch.org/whl/cpu
   pip install transformers==4.37.0

### Initiating Hugging Face CLI for access to model

4. **Install the Hugging Face hub CLI:**
   ```
   !python3 -m pip install -U "huggingface_hub[cli]"
5. **Run the notebook login:**
   ```
   from huggingface_hub import notebook_login
   notebook_login()
   # Enter your Hugging Face Access Token in the Text Form
6. **Once the UI loaded, enter the Access Token you copied earlier from the Hugging Face setup and click "Login".**
7. **Now you are logged in as an authorized user for the ML model you requested.**



### A few more setups
8. Enter in a new cell the following import statements.
   ```
   !pip install torch
   import torch
   import json
   import os
### Using the models
9. Generate response.
```
   import transformers
   from transformers import AutoTokenizer, AutoModelForCausalLM
   
   # Define model name and token directly
   model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
   HUGGINGFACE_TOKEN = '<your-huggingface-access-token-here>' 
   
   tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, token=HUGGINGFACE_TOKEN)
   model = AutoModelForCausalLM.from_pretrained(
       model_name, 
       trust_remote_code=True, 
       token=HUGGINGFACE_TOKEN
   )
   
   # Add a padding token if not already present
   if tokenizer.pad_token is None:
       tokenizer.add_special_tokens({'pad_token': tokenizer.eos_token})
       model.resize_token_embeddings(len(tokenizer))
   
   # Move model to CPU
   device = torch.device("cpu")
   model.to(device)
   
   # Test the model
   prompt = "What is Meta Llama 3?"
   inputs = tokenizer(prompt, return_tensors="pt", padding=True).to(device)
   
   # Provide attention mask and handle pad token id
   attention_mask = inputs["attention_mask"].to(device)
   outputs = model.generate(
       inputs["input_ids"], 
       attention_mask=attention_mask, 
       max_length=50, 
       pad_token_id=tokenizer.eos_token_id
   )
   response = tokenizer.decode(outputs[0], skip_special_tokens=True)
   print(response)
```

### Refining the output of the models
10. Refine the output.

Key Observations:
* Question is repeated in the response
* Response does not end in a full sentence
```
   import transformers
   from transformers import AutoTokenizer, AutoModelForCausalLM
   
   # Define model name and token directly
   model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
   HUGGINGFACE_TOKEN = '<your-huggingface-access-token-here>' 
   
   tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, token=HUGGINGFACE_TOKEN)
   model = AutoModelForCausalLM.from_pretrained(
       model_name, 
       trust_remote_code=True, 
       token=HUGGINGFACE_TOKEN
   )
   
   # Add a padding token if not already present
   if tokenizer.pad_token is None:
       tokenizer.add_special_tokens({'pad_token': tokenizer.eos_token})
       model.resize_token_embeddings(len(tokenizer))
   
   # Move model to CPU
   device = torch.device("cpu")
   model.to(device)
   
   #Test the model
   def generate_response(prompt):
       inputs = tokenizer(prompt, return_tensors="pt", padding=True).to(device)
       attention_mask = inputs["attention_mask"].to(device)
       outputs = model.generate(
           inputs["input_ids"], 
           attention_mask=attention_mask, 
           max_length=50, 
           pad_token_id=tokenizer.eos_token_id
       )
       response = tokenizer.decode(outputs[0], skip_special_tokens=True)
       return response
   
   # First prompt
   prompt1 = "What is Meta LLama 3"
   response1 = generate_response(prompt1)
   print("Prompt 1:\n", prompt1)
   print("Response 1:\n", response1)
   
   print(" ")
   
   # Second prompt
   prompt2 = "What is Intel IPEX LLM"
   response2 = generate_response(prompt2)
   print("Prompt 2:\n", prompt2)
   print("Response 2:\n", response2)
```

### Further refinement
11. Adjust with more specific parameters

Key Observations: 
* Question is now no longer repeated.
* Response still doesn't end in a full sentence. 
```
   import torch
   import transformers
   from transformers import AutoTokenizer, AutoModelForCausalLM
   
   # Define model name and token directly
   model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
   HUGGINGFACE_TOKEN = '<your-huggingface-access-token-here>' 
   
   # Initialize tokenizer and model
   tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, token=HUGGINGFACE_TOKEN)
   model = AutoModelForCausalLM.from_pretrained(
       model_name, 
       trust_remote_code=True, 
       token=HUGGINGFACE_TOKEN
   )
   
   # Add a padding token if not already present
   if tokenizer.pad_token is None:
       tokenizer.add_special_tokens({'pad_token': tokenizer.eos_token})
       model.resize_token_embeddings(len(tokenizer))
   
   # Move model to CPU
   device = torch.device("cpu")
   model.to(device)
   
   def generate_response(prompt, max_length=100):
       inputs = tokenizer(prompt, return_tensors="pt", padding=True).to(device)
       attention_mask = inputs["attention_mask"].to(device)
       outputs = model.generate(
           inputs["input_ids"], 
           attention_mask=attention_mask, 
           max_length=max_length, 
           pad_token_id=tokenizer.eos_token_id,
           eos_token_id=tokenizer.eos_token_id,
           do_sample=True,
           top_k=50,
           top_p=0.95,
       )
       response = tokenizer.decode(outputs[0], skip_special_tokens=True)
       return response
   
   # Test the model with two prompts
   prompt1 = "What is Meta LLama 3"
   response1 = generate_response(prompt1)
   print("Prompt 1:", prompt1)
   print("Response 1:", response1)
   
   print(" ")
   
   prompt2 = "What is Intel IPEX LLM"
   response2 = generate_response(prompt2)
   print("Prompt 2:", prompt2)
   print("Response 2:", response2)
```

### Final refinement
12. Tune response to end in full sentence.

Key Observations:
* Now Question is not repeated in the response.
* Now sentence ends in a full sentence.
```
   import torch
   import transformers
   from transformers import AutoTokenizer, AutoModelForCausalLM
   
   # Define model name and token directly
   model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
   HUGGINGFACE_TOKEN = '<your-huggingface-access-token-here>' 
   
   # Initialize tokenizer and model
   tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, token=HUGGINGFACE_TOKEN)
   model = AutoModelForCausalLM.from_pretrained(
       model_name, 
       trust_remote_code=True, 
       token=HUGGINGFACE_TOKEN
   )
   
   # Add a padding token if not already present
   if tokenizer.pad_token is None:
       tokenizer.add_special_tokens({'pad_token': tokenizer.eos_token})
       model.resize_token_embeddings(len(tokenizer))
   
   # Move model to CPU
   device = torch.device("cpu")
   model.to(device)
   
   def generate_response(prompt, max_length=100):
       inputs = tokenizer(prompt, return_tensors="pt", padding=True).to(device)
       attention_mask = inputs["attention_mask"].to(device)
       outputs = model.generate(
           inputs["input_ids"], 
           attention_mask=attention_mask, 
           max_length=max_length, 
           pad_token_id=tokenizer.eos_token_id,
           eos_token_id=tokenizer.eos_token_id,
           do_sample=True,
           top_k=50,
           top_p=0.95,
       )
       response = tokenizer.decode(outputs[0], skip_special_tokens=True)
       
       # Ensure the response ends with a complete sentence
       end_punctuation = {'.', '!', '?'}
       for i in range(len(response)-1, 0, -1):
           if response[i] in end_punctuation:
               return response[:i+1]
       
       return response
   
   # Test the model with two prompts
   prompt1 = "What is Meta LLama 3"
   response1 = generate_response(prompt1)
   print("Prompt 1:", prompt1)
   print("Response 1:", response1)
   
   print(" ")
   
   prompt2 = "What is Intel IPEX LLM"
   response2 = generate_response(prompt2)
   print("Prompt 2:", prompt2)
   print("Response 2:", response2)
```


## Conclusion
Through the development of this Generative AI application, leveraging the Intel Tiber Developer Cloud, I successfully integrated the Meta Llama 3 model with the Intel® Extension for PyTorch (IPEX). This project demonstrates the effectiveness of combining state-of-the-art machine learning models with advanced hardware acceleration to achieve significant performance improvements. The comprehensive setup and iterative refinements ensured robust functionality and efficient model inference, providing valuable insights and practical experience in optimizing AI workloads.


## Tech Stack
- **Intel Tiber Developer Cloud:** Platform for development and execution
- **Jupyter Notebooks:** Interactive environment for code execution and visualization
- **Meta Llama 3 Model:** State-of-the-art generative AI model
- **Intel® Extension for PyTorch (IPEX):** Tool for optimizing PyTorch models
- **Python:** Primary programming language
- **Conda:** Environment management
- **Hugging Face:** Platform for accessing and managing ML models
- **PyTorch:** Deep learning framework
