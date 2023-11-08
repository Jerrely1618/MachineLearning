from sentence_transformers import SentenceTransformer, util
from icecream import ic
import torch
import time

SEARCH = 'fulltime'
FILTERS= ['Full-Time', 'Flexible', 'Remote', 'Part-Time','Processing products','License required','Consistent','part-time','where you are from']

@time
def matchFilters(search: str, filters: []) -> str:
    model = SentenceTransformer('all-mpnet-base-v2')
    embs_a = torch.zeros(1, 768)
    embs_b = torch.zeros(len(filters), 768)

    embedding_word_a = model.encode(search, convert_to_tensor = True)
    embs_a[0] = embedding_word_a
    for i, word_b in enumerate(filters):
        embedding_word_b = model.encode(word_b, convert_to_tensor = True)
        embs_b[i] = embedding_word_b

    idx = torch.max(util.cos_sim(embs_a, embs_b), dim=-1).indices
    found_word = filters[idx]

    return found_word

matchFilters(SEARCH,FILTERS)