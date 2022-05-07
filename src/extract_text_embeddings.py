import re

import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
from transformers import DistilBertTokenizer, DistilBertModel

from globals import movie_text_info_file, text_embeddings_file_path, item_metadata_file


def _prepare_text(title, plot, tags) -> str:
    # remove text in parentheses and add [SEP] tokens for different sentences
    processed_plot = re.sub(r'(\.|\?|!)', ' [SEP] ', re.sub(r'\([^)]*\)', '', plot)).rsplit('[SEP]', 1)[0]
    # concat title and processed plot
    # TODO: add tags?
    return title + ' [SEP] ' + processed_plot


def _sentence_emb_from_word_emb(word_embs, use_avg_instead_of_cls=False):
    """
    Takes in padded (sequence_length, emb_size) tensor for word embeddings in a
    single sequence and aggregates them in dim=0 (i.e. over all words in sequence)
    to get a (emb_size) 1-d tensor with a sequence-level embedding.
    """
    # related dilemma:
    # https://stackoverflow.com/questions/62705268/why-bert-transformer-uses-cls-token-for-classification-instead-of-average-over
    if use_avg_instead_of_cls:
        return torch.mean(word_embs, dim=0)  # take a simple average
    else:
        return word_embs[0, :]  # take only the first token's embedding (i.e. the [CLS] token)


def extract_text_features(use_avg_instead_of_cls):
    # load features given
    movie_info = pd.read_csv(movie_text_info_file + '.csv', index_col=0)

    # run titles though a pretrained transformer to get contextual word embeddings for each word in the ad
    """ BERT provides us with contextual bidirectional embeddings for words. DistilBERT is a lighter version of BERT.
    theory: https://becominghuman.ai/bert-transformers-how-do-they-work-cd44e8e31359
    source: https://huggingface.co/distilbert-base-uncased
    """
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased', max_length=512, truncation=True)
    transformer: DistilBertModel = DistilBertModel.from_pretrained("distilbert-base-uncased")
    transformer.eval()

    # save embeddings to a numpy array progressively
    # I do it one-by-one which might be less efficient but we only do this once
    embeddings = np.zeros((movie_info.shape[0], 768))
    with torch.no_grad():
        for i, (_, data) in tqdm(enumerate(movie_info.iterrows()), total=movie_info.shape[0], desc='Creating text embeddings'):
            # encode title
            while True:
                try:
                    text = _prepare_text(data['primaryTitle'], data['plot'], data['tags'])
                    encoded_text = tokenizer(text, return_tensors='pt')
                    # pass through model
                    out = transformer(**encoded_text)['last_hidden_state']
                    break
                except RuntimeError:
                    print("Warning: found too large a sentence, repeating without the last sentence.")
                    data['plot'] = data['plot'].rsplit('.', 1)[0]  # remove sentences until it fits
            out = out.squeeze(0)  # no batch size here
            # get one embedding for the whole title
            emb = _sentence_emb_from_word_emb(out, use_avg_instead_of_cls=use_avg_instead_of_cls)
            # put it numpy array
            embeddings[i, :] = emb.numpy()

    # create dataframe with corresponding keys (i.e. cc column)
    text_embeddings = pd.DataFrame(index=movie_info.index, data=embeddings, columns=[f'text_{i+1}' for i in range(embeddings.shape[1])])
    print(text_embeddings)

    # save as hdf because it means that we load what we save exactly with no pesky csv problems
    text_embeddings.to_hdf(text_embeddings_file_path + '.h5', key='title_embeddings', mode='w')


if __name__ == '__main__':
    # extract embeddings from text features (titles) and save them
    extract_text_features(use_avg_instead_of_cls=True)

    # load and print saved dataframe for sanity check
    test: pd.DataFrame = pd.read_hdf(text_embeddings_file_path + '.h5', key='title_embeddings')
    test.rename(columns={i: f'text_{i+1}' for i in range(768)}, inplace=True)
    print(test)

    # load metadata and concat
    metadata: pd.DataFrame = pd.read_hdf('../data/item_metadata_backup_before_text' + '.h5')
    metadata = pd.concat([metadata, test], axis=1)
    print(metadata)
    metadata.to_hdf(item_metadata_file + '.h5', key='metadata', mode='w')
