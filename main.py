import os.path as osp

import fasttext
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from tqdm import tqdm

import argparse


FASTTEXT_MODEL_PATH = "~/store/cc.en.300.bin"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ft_model_path",
        default=osp.expanduser("data/cc.en.300.bin"),
        action="store",
        type=str,
        help="Path to pre-trained ft model.",
    )
    parser.add_argument(
        "--path_entity_types",
        default="data/entity_types.parquet",
        type=str,
        action="store",
        help="Path to file with entity types.",
    )
    parser.add_argument(
        "--path_entities",
        default="data/yago_entities.csv",
        type=str,
        action="store",
        help="Path to output file that holds only entities.",
    )

    parser.add_argument(
        "--n_dimensions",
        default=300,
        type=int,
        action="store",
        help="Output dimensions. If < 300, dimensions will be reduced accordingly.",
    )

    args = parser.parse_args()

    return args


def extract_entities(path_to_entity_types, path_to_entities=None):
    # Reading pq file with KEN embeddings
    tab = pq.read_table(path_to_embs)
    entities = tab.to_pandas()

    # Saving entities list to csv
    if path_to_entities is not None:
        entities.to_csv(path_to_entities, index=False)
    return entities    


def generate_embeddings(entities, output_emb_path, n_dimensions=300):
    # Making sure that the datatype is what we're expecting
    assert type(entities) == pd.DataFrame
    
    # Loading fasttext model
    model = _load_model(n_dimensions)

    # Filling np.array with the entity embeddings
    emb_array = _populate_array(entities, model, n_dimensions)

    # Converting the array to df
    df = convert_array_to_df(entities, emb_array) 

    print("Saving file to parquet.")
    df.to_parquet(output_emb_path)


def _populate_array(entities, model, n_dimensions):
    print("Creating table")
    emb_array = np.zeros((len(entities), n_dimensions))

    for _, e in tqdm(entities.iterrows(), total=len(entities)):
        e_key = e["Entity"]
        # Remove symbols to avoid noise in get_sentence_vector
        entity_str = e_key.replace("<", "").replace(">", "").replace("_", " ")
        entity_emb =  model.get_sentence_vector(entity_str)
        emb_array[_, :] = entity_emb
    
    return emb_array

def convert_array_to_df(entities, emb_array):
    print("Converting np.array to pd.DataFrame")
    df = pd.DataFrame(emb_array, index=entities["Entity"])
    df.columns = [f'X{_}' for _ in df.columns]  # type: ignore
    return df
    

def _load_model(n_dimensions=300):
    print("Loading fasttext model.")
    model_path = osp.expanduser(FASTTEXT_MODEL_PATH)
    assert osp.exists(model_path)
    
    model = fasttext.load_model(model_path)

    if n_dimensions < 300:
        print(f"Reducing model size to {n_dimensions}.")
        model = fasttext.util.reduce_model(model, n_dimensions)
        return model
    elif n_dimensions == 300:
        return model
    else:
        raise ValueError(f"Number of dimensions {n_dimensions} is larger than 300.")


if __name__ == "__main__":
    args = parse_args()
    entity_types_path = args.path_entity_types
    assert osp.exists(entity_types_path)

    path_to_entities = args.path_entities

    if not osp.exists(path_to_entities):
        print("Extracting entities.")
        entities = extract_entities(path_to_embs, path_to_entities)
    else:
        print("Loading entities.")
        entities = pd.read_csv(path_to_entities)

    n_dimensions = args.n_dimensions

    output_emb_path = f"data/yago3-fasttext.{n_dimensions}.parquet"
    generate_embeddings(entities, output_emb_path, n_dimensions)
    