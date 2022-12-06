import os.path as osp

import fasttext
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from tqdm import tqdm

FASTTEXT_MODEL_PATH = "~/store/cc.en.300.bin"


def extract_entities(path_to_embs, path_to_entities=None):
    # Reading pq file with KEN embeddings
    tab = pq.read_table(path_to_embs)
    entities = tab.to_pandas()

    # Saving entities list to csv
    if path_to_entities is not None:
        entities.to_csv(path_to_entities, index=False)
    return entities    

def generate_embeddings(entities, output_emb_path):
    model = load_model()
    
    assert type(entities) == pd.DataFrame
    emb_array = populate_array(entities, model)
 
    df = convert_array_to_df(entities, emb_array) 

    print("Saving file to parquet.")
    df.to_parquet(output_emb_path)

def populate_array(entities, model):
    print("Creating table")
    emb_array = np.zeros((len(entities), 300))
    for _, e in tqdm(entities.iterrows(), total=len(entities)):
        e_key = e["Entity"]
        entity_str = e_key.replace("<", "").replace(">", "").replace("_", " ")
        entity_emb =  model.get_sentence_vector(entity_str)
        emb_array[_, :] = entity_emb
    return emb_array

def convert_array_to_df(entities, emb_array):
    print("Converting np.array to pd.DataFrame")
    df = pd.DataFrame(emb_array, index=entities["Entity"])
    df.columns = [f'X{_}' for _ in df.columns]  # type: ignore
    return df
    
def load_model():
    model_path = osp.expanduser(FASTTEXT_MODEL_PATH)
    assert osp.exists(model_path)
    
    model = fasttext.load_model(model_path)
    return model # type: ignore
    

if __name__ == "__main__":
    path_to_embs = osp.expanduser("~/store/emb_mure_ken_yago3_full.parquet")
    assert osp.exists(path_to_embs)

    path_to_entities = "data/yago_entities.csv"

    if not osp.exists(path_to_entities):
        print("Extracting entities.")
        entities = extract_entities(path_to_embs, path_to_entities)
    else:
        print("Loading entities.")
        entities = pd.read_csv(path_to_entities)
    output_emb_path = "data/yago3-fasttext.parquet"
    generate_embeddings(entities, output_emb_path)

    