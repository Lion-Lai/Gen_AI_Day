#%%
import tiktoken as t
encoding = t.encoding_for_model("gpt-3.5-turbo")
tokens = encoding.encode("jqhrkl3hrjlhjkkldjfklsjdfke")
#%%
decoded = " ".join(encoding.decode([token]) for token in tokens)
print(decoded)
# %%

    
# %%
