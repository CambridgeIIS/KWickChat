import configparser
import torch
from sentence_transformers import SentenceTransformer
import numpy as np
from utils import *
from eval_utils import *
from nltk.translate.bleu_score import sentence_bleu
import re

def num_key_words(encoded_dataset,tokenizer,embeddings,embedding_file,MetricWrapper,DEFAULT,model):
    vector_average = MetricWrapper.factory('vector_average')
    vector_extrema = MetricWrapper.factory('vector_extrema')
    greedy_matching = MetricWrapper.factory('greedy_matching')

    results = {"train": [], "valid": []}
    model_sentence_transformer = SentenceTransformer('/home/shawn/desktop/transfer-learning-conv-ai-master')
    score_wer_list_list = []
    score_bleu_list_list = []
    vector_average_score_list_list = []
    vector_extrema_score_list_list = []
    greedy_matching_score_list_list = []

    range_x = np.arange(1, 10)
    file1 = open("results/num_key_words.txt", "w")

    for number_words in range_x:
        score_bleu_list = []
        score_wer_list = []
        vector_average_score_list = []
        vector_extrema_score_list = []
        greedy_matching_score_list = []

        file1.write('=========================================\n')
        file1.write('=========================================\n')
        file1.write('==================={}==================\n'.format(number_words))
        file1.write('=========================================\n')
        file1.write('=========================================\n')

        for dataset_name, dataset in encoded_dataset.items():
            if dataset_name == 'train':
                continue
            for i, dialog in enumerate(dataset):
                #             if i >10:
                #                 continue

                personality = dialog['personality']
                #             for j, utterance in enumerate(dialog['utterances']):
                utterance = dialog['utterances'][-1]
                # for k, history in enumerate(utterance['candidate']):
                reply = utterance['candidates'][-1]
                golden_reply = tokenizer.decode(reply, skip_special_tokens=True)

                history = utterance['history']

                try:
                    keyphrase = keyphrase_extract(golden_reply, model_sentence_transformer, number_words)
                except ValueError:
                    keyphrase = golden_reply.split(' ')[:number_words]

                key_phrase = [tokenizer.encode(keyphrase[0])]

                with torch.no_grad():
                    out_ids = sample_sequence(list(personality), list(history), tokenizer, key_phrase, model, DEFAULT)

                predicted_reply = tokenizer.decode(out_ids, skip_special_tokens=True)

                reference = re.findall(r"[\w']+|[.,!?;]", golden_reply)
                candidate = re.findall(r"[\w']+|[.,!?;]", predicted_reply)

                reference_word = [word for word in reference]
                candidate_word = [word for word in candidate]

                vector_average_score = vector_average.eval(
                    h=predicted_reply,
                    r=golden_reply,
                    embeddings=embeddings,
                    embedding_file=embedding_file
                )
                vector_extrema_score = vector_extrema.eval(
                    h=predicted_reply,
                    r=golden_reply,
                    embeddings=embeddings,
                    embedding_file=embedding_file
                )
                greedy_matching_score = greedy_matching.eval(
                    h=predicted_reply,
                    r=golden_reply,
                    embeddings=embeddings,
                    embedding_file=embedding_file
                )

                score_bleu = sentence_bleu([reference], candidate)
                score_wer = lev_dist(reference_word, candidate_word) / len(reference_word)

                print('golden reply: {}'.format(golden_reply))
                print('key phrases: {}'.format(keyphrase))
                print('predicted reply: {}'.format(predicted_reply))
                print('BLEU score: {}\n'.format(score_bleu))
                print('WER score: {}\n'.format(score_wer))
                print('vector_average_score: {}\n'.format(vector_average_score))
                print('vector_extrema_score: {}\n'.format(vector_extrema_score))
                print('greedy_matching_score: {}\n'.format(greedy_matching_score))

                print('\n')

                file1.write('golden reply: {}\n'.format(golden_reply))
                file1.write('key phrases: {}\n'.format(keyphrase))
                file1.write('predicted reply: {}\n'.format(predicted_reply))
                file1.write('BLEU score: {}\n'.format(score_bleu))
                file1.write('WER score: {}\n'.format(score_wer))
                file1.write('vector_average_score: {}\n'.format(vector_average_score))
                file1.write('vector_extrema_score: {}\n'.format(vector_extrema_score))
                file1.write('greedy_matching_score: {}\n'.format(greedy_matching_score))

                file1.write('\n')

                score_bleu_list.append(score_bleu)
                score_wer_list.append(score_wer)
                vector_average_score_list.append(vector_average_score)
                vector_extrema_score_list.append(vector_extrema_score)
                greedy_matching_score_list.append(greedy_matching_score)

        score_bleu_list_list.append(score_bleu_list)
        score_wer_list_list.append(score_wer_list)
        vector_average_score_list_list.append(vector_average_score_list)
        vector_extrema_score_list_list.append(vector_extrema_score_list)
        greedy_matching_score_list_list.append(greedy_matching_score_list)
    #             results[dataset_name][i]['utterances'][j]['result'] = result
    file1.close()
    np.savez('results/num_key_words', score_bleu_list_list, score_wer_list_list, \
             vector_average_score_list_list, vector_extrema_score_list_list, greedy_matching_score_list_list)