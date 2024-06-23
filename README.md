# Meta-Llama3-Intel-IPEX-LLM

## Overview

This repository contains a Jupyter Notebook for experimenting with the Meta-Llama3 model and Intel IPEX LLM. The provided notebook allows you to run queries and generate responses using the specified models. Below, you'll find detailed steps to set up your environment, run the notebook, and interpret the results.

## Setup Instructions

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

## Running The Notebook

1. **Open the Jupyter Notebook:**
   Launch Jupyter Lab or Jupyter Notebook and open the Meta-Llama3-Intel-IPEX-LLM.ipynb file.

2. **Login to Hugging Face:**
   The notebook will prompt you to enter your Hugging Face token. Follow the instructions to log in.

3. **Generate Responses:**
   Run the cells in the notebook to generate responses for the provided prompts. The notebook contains code to initialize the models, generate responses, and ensure the responses end in complete sentences.
