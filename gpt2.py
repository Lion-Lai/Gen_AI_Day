#%%
from transformers import pipeline, set_seed
generator = pipeline('text-generation', model='gpt2')
set_seed(42)
generator("I like eat apple and ", max_length=30, num_return_sequences=5)

# %%
