import os
import numpy as np
import clip

device = "cuda:0"
prompt = 'An H&E image patch of [] tissue.'.replace('[]', 'malignant')

model, preprocess = clip.load("ViT-B/32", device=device)
model.eval()
idx = clip.tokenize(prompt, truncate=True).to(device)
text_fea = model.encode_text(idx).detach().cpu().numpy()

print(text_fea)