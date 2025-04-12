#%%
import torch
#text = "Just another Kusto hacker."
torch.set_default_dtype(torch.double)
torch.set_printoptions(precision=20, sci_mode=False)

#%%
ord("a"), ord("z"), ord("A"), ord("Z")
torch.concat(
    (torch.arange(ord("a"), ord("z")+1), torch.arange(ord("A"), ord("Z")+1),
     torch.arange(ord(" "), ord(" ")+1)
    )
)
#%%
#letter_ascii_codes = torch.arange(32, 123)
letter_ascii_codes=torch.concat(
    (torch.arange(ord("a"), ord("z")+1), torch.arange(ord("A"), ord("Z")+1),
     torch.arange(ord(" "), ord(" ")+1)
    )
)
total_letters = len(letter_ascii_codes)
print( total_letters)
one_hot_encoded = torch.eye(total_letters)
one_hot_encoded

#%%
# token is the index of the letter in the letters tensor
def GetTokenFromChar(char):
    return torch.where(letter_ascii_codes == ord(char))[0].item()  # Get the index of the character in the letters tensor
#ord(char) - letter_ascii_codes[0]
ord("a"), GetTokenFromChar("a"),ord("A"), GetTokenFromChar("A"), ord(" "),GetTokenFromChar(" ")  # Should return the index of ' ' in the letters tensor

#%%
def GetTokensFromText(text):
    return torch.tensor([GetTokenFromChar(char) for char in text])  # Convert to tensor and add a dimension 
tokens = GetTokensFromText("Ju")

[ chr(ascii_code) for ascii_code in letter_ascii_codes[tokens] ]

#%%
def GetCharFromProbs(logits):
    return chr(letter_ascii_codes[logits.argmax(dim=-1).item()].item())
GetCharFromProbs( one_hot_encoded[2] )  
#%%
def GetProbsFromChar(char):
    return one_hot_encoded[GetTokenFromChar(char)]
GetProbsFromChar("J")  # Should return the one-hot encoded vector for 'J'
#%%
def GetTrainingDataFromText(text):
    prompt = text[:-1]
    prediction = text[-1]
    X = GetTokensFromText(prompt)
    Y = GetProbsFromChar(prediction)
    return X, Y

X, Y = GetTrainingDataFromText("Jus")
X, Y
#%%
import torch
import torch.nn as nn
torch.manual_seed(123)  # Set random seed for reproducibility

class LanguageModel(nn.Module):
    def __init__(self, dimensions :int = 64, total_letters :int = 91, embedding_dim :int = 1):
        D = dimensions
        super(LanguageModel, self).__init__()

        self.emebedding_matrix = torch.randn(total_letters, embedding_dim)  # Random embedding matrix
        
        self.Q = nn.Linear(in_features = embedding_dim, out_features = D, bias = False)  # Q matrix for the linear transformation
        self.K = nn.Linear(in_features = embedding_dim, out_features =D, bias = False)
        self.V = nn.Linear(in_features = embedding_dim, out_features =D, bias = False)

        # self.MLP = nn.Sequential(
        #         nn.Linear(in_features = D, out_features = D),
        #         nn.ReLU(),
        #         nn.Linear(in_features = D, out_features = D),
        #      )
        
        self.O = nn.Linear(in_features = D, out_features = total_letters, bias = False)

    def forward(self, x): # x: [T,1]
        x = self.emebedding_matrix[x]  # [T,1] @ [1, total_letters] = [T, embedding_dim]
        q = self.Q(x) # [T,1] @ [1, D] = [T, D]
        k = self.K(x) # [T,1] @ [1, D] = [T, D]
        v = self.V(x) # [T,1] @ [1, D] = [T, D]

        #attentionMatrix = q @ k.T # [T, D] @ [D, T] = [T, T]
        #attentionMatrix = (q @ k.T) / (q.size(-1) ** 0.5)  # Normalize by sqrt(D)
        attentionMatrix = (q @ k.T) / (q.size(-1) ** 0.5)  # Normalize by sqrt(D)
        attentionMatrix = attentionMatrix.softmax(dim=-1) # [T, T]

        x = attentionMatrix @ v # [T, T] @ [T, D] = [T, D]

        #x = self.MLP(x) # [T, D] @ [D, D] = [T, D]
        x = self.O(x) # [T, D] @ [D, total_letters] = [T, total_letters]
        return x[-1]  # Fix: Return logits for the last character

lm = LanguageModel(dimensions = 4, total_letters = total_letters, embedding_dim = 5)
X, Y = GetTrainingDataFromText("Jus")
probs = lm(X)  # Add a dimension to X
X.shape, X.shape, Y.shape, probs

#%%
def count_parameters(model):
    return sum(p.numel() for p in model.parameters())

# Example usage
model = lm
total_params = count_parameters(model)
print(f"Total parameters: {total_params}")
# %%
#%%
def GenerateText(lm, start_text, num_chars):
    lm.eval()  # Ensure the model is in evaluation mode
    generated_text = start_text
    for _ in range(num_chars):
        X = GetTokensFromText(generated_text)
        logits = lm(X)
        next_char = GetCharFromProbs(logits)
        generated_text += next_char
    return generated_text

GenerateText(lm, "J", 24)  # Generate 10 characters starting with "J"
#%%
lm.train()
optimizer = torch.optim.Adam(lm.parameters(), lr=0.01)
loss_function = nn.CrossEntropyLoss()
text = "Just another Kusto hacker "
generated_text = None
def training_loop():
    for episode in range(100000):
        total_loss = 0  # Track total loss for logging
        for i in range(2, len(text) + 1):
            current_text = text[0:i]
            X, Y = GetTrainingDataFromText(current_text)

            logits = lm(X)
            loss = loss_function(logits, Y)
            loss.backward()
            total_loss += loss           
            optimizer.step()
            optimizer.zero_grad()
        if episode % 10 == 0:
            print(f"Episode {episode}: Total Loss = {total_loss:.4f}")
            lm.eval()
            generated_text = GenerateText(lm, "J", 24)
            print( generated_text )
            lm.train()
        if total_loss < 0.1:  # Fix: Use total loss for termination
            return
        if generated_text == text[:-1] and total_loss < 0.2:
            print("Training complete.")
            return
training_loop()
# %%
lm.eval()

xx = GetTokensFromText("J")
lm.emebedding_matrix[xx]
#%%
def count_parameters(model):
    return sum(p.numel() for p in model.parameters())

# Example usage
model = lm
total_params = count_parameters(model)
print(f"Total parameters: {total_params}")
# %%
GenerateText(lm, "J", 24)
# %%
#################### dump
lm.eval()
torch.set_printoptions(precision=20, sci_mode=False)
lines = []
for i, m in enumerate( lm.emebedding_matrix):
    lines.append(f"    {i + 1}, dynamic({[e.item() for e in m]}),\r\n")
lines[-1] = lines[-1].replace("),\r\n", ")\r\n")
with open("output\\parameters.txt", "w") as f:
    f.writelines("let GPT_EmbeddingMatrix =  ( datatable (row_id: long, vector:dynamic)[")
    f.writelines(lines)
    f.writelines("]);\r\n")

    lines = []
    ts = [p for p in lm.Q.parameters()][0].T
    for i, m in enumerate( ts ):
        lines.append(f"    {i + 1}, dynamic({[e.item() for e in m]}),\r\n")
    lines[-1] = lines[-1].replace("),\r\n", ")\r\n")
    f.writelines("let GPT_QueryMatrix =  ( datatable (row_id: long, vector:dynamic)[")
    f.writelines(lines)
    f.writelines("]);\r\n")

    lines = []
    ts = [p for p in lm.K.parameters()][0].T
    for i, m in enumerate( ts ):
        lines.append(f"    {i + 1}, dynamic({[e.item() for e in m]}),\r\n")
    lines[-1] = lines[-1].replace("),\r\n", ")\r\n")
    f.writelines("let GPT_KeyMatrix =  (  datatable (row_id: long, vector:dynamic)[")
    f.writelines(lines)
    f.writelines("]);\r\n")


    lines = []
    ts = [p for p in lm.V.parameters()][0].T
    for i, m in enumerate( ts ):
        lines.append(f"    {i + 1}, dynamic({[e.item() for e in m]}),\r\n")
    lines[-1] = lines[-1].replace("),\r\n", ")\r\n")
    f.writelines("let GPT_ValueMatrix =  (  datatable (row_id: long, vector:dynamic)[")
    f.writelines(lines)
    f.writelines("]);\r\n")


    lines = []
    ts = [p for p in lm.O.parameters()][0].T
    for i, m in enumerate( ts ):
        lines.append(f"    {i + 1}, dynamic({[e.item() for e in m]}),\r\n")
    lines[-1] = lines[-1].replace("),\r\n", ")\r\n")
    f.writelines("let GPT_OutputMatrix =  (  datatable (row_id: long, vector:dynamic)[")
    f.writelines(lines)
    f.writelines("]);\r\n")

# %%
torch.set_printoptions(precision=20, sci_mode=False)
xxx = GetTokensFromText("J")
xx = lm.emebedding_matrix[xxx]  # [T,1] @ [1, total_letters] = [T, embedding_dim]
q = lm.Q(xx) # [T,1] @ [1, D] = [T, D]

k = lm.K(xx) # [T,1] @ [1, D] = [T, D]
v = lm.V(xx) # [T,1] @ [1, D] = [T, D]
v

attentionMatrix = (q @ k.T) / (q.size(-1) ** 0.5)  # Normalize by sqrt(D)

#%%
attentionMatrix = attentionMatrix.softmax(dim=-1) # [T, T]
attentionMatrix
#%%
a = attentionMatrix @ v # [T, T] @ [T, D] = [T, D]
v
#%%

out = lm.O(a) # [T, D] @ [D, total_letters] = [T, total_letters]
out
#%%
out[-1]  # Fix: Return logits for the last character

#lm.emebedding_matrix[xx]
# %%
