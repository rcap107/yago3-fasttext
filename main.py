import pyarrow.parquet as pq
import os.path as osp
import fasttext
import os.path as osp
import fasttext
from joblib import Parallel, delayed
import pandas as pd
from tqdm import tqdm
import pickle

FASTTEXT_MODEL_PATH = "~/store/cc.en.300.bin"


def extract_entities(path_to_embs, path_to_entities=None):
    # Reading pq file with KEN embeddings
    tab = pq.read_table(path_to_embs)
    df = tab.to_pandas()
    entities = df["Entity"]
    # Cleaning entity names
    entities = entities.str.replace("<", "").str.replace(">", "").str.replace("_", " ")

    # Saving entities list to csv
    if path_to_entities is not None:
        entities.to_csv(path_to_entities, index=False)
    return entities    

def generate_embeddings(entities, output_emb_path):
    model_path = osp.expanduser(FASTTEXT_MODEL_PATH)
    assert osp.exists(model_path)

    model = fasttext.load_model(model_path)
    
    emb_dict = {}
    for _, e in tqdm(entities.iterrows(), total=len(entities)):
        entity_str = e["Entity"].lower()
        entity_emb =  model.get_sentence_vector(entity_str)
        emb_dict[entity_str] = entity_emb

    # This takes a very long time to run! 
    print("Converting dict to pd.DataFrame")
    df = pd.DataFrame().from_dict(emb_dict, orient="index")
    df.columns = [f'X{_}' for _ in df.columns]  # type: ignore
    print("Saving df to parquet file.")
    df.reset_index().to_parquet(f"{output_emb_path}")


if __name__ == "__main__":
    path_to_embs = osp.expanduser("~/store/emb_mure_ken_yago3_full.parquet")
    assert osp.exists(path_to_embs)
    path_to_entities = "data/yago_entities.csv"
    entities = extract_entities(path_to_embs, path_to_entities)
    output_emb_path = "data/yago3-fasttext.parquet"
    generate_embeddings(entities, output_emb_path)

    