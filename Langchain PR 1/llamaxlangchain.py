# -*- coding: utf-8 -*-


#nvidia-smi

#pip install -q transformers einops accelerate langchain bitsandbytes

"""## Loggin hugging face account"""

#huggingface-cli login

"""## Import all required liblaries"""

from langchain.llms import HuggingFacePipeline
from transformers import AutoTokenizer, pipeline
import torch
import warnings

warnings.filterwarnings("ignore")

#!pip install -q  langchain-community

"""## LOAD LLAMA 2 MODEL"""

# define model to use
model ="meta-llama/Meta-Llama-3.1-8B-Instruct"
#model= "meta-llama/Llama-2-7b-chat"

tokenizer = AutoTokenizer.from_pretrained(model)

pipeline = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    model_kwargs={"torch_dtype": torch.bfloat16},
    trust_remote_code = True ,
    device_map="auto",
    max_length=1000,
    do_sample=True,
    top_k=10,
    num_return_sequences=1,
    eos_token_id=tokenizer.eos_token_id
)

llm = HuggingFacePipeline(pipeline=pipeline, model_kwargs={"temperature": 0})

prompt = "What would be a cool name and catchy for a software engineering company and get inspiration from effects of weed"

print(llm(prompt))

"""## Custom prompt"""

from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

prompt_tem = PromptTemplate(
    input_variables=["cuisine"],
    template="I want ot open a resturante for {cuisine} food.Suggest an fancy name for this "
                            )

input_prompt = prompt_tem.format(cuisine="Italian")

print(input_prompt)

chain = LLMChain(llm=llm, prompt=prompt_tem, verbose=True)
response = chain.run("italian")
print(response)