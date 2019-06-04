import os
import sys
import json
import pickle

import nltk
import tqdm
from torchvision import transforms
from PIL import Image
from transforms import Scale

def process_question(root, out_dir, split, word_dic=None, answer_dic=None):
    if word_dic is None:
        word_dic = {}

    if answer_dic is None:
        answer_dic = {}

    with open(os.path.join(root, 'questions',
                        f'CLEVR_{split}_questions.json')) as f:
        data = json.load(f)

    result = []
    word_index = 1
    answer_index = 0

    for question in tqdm.tqdm(data['questions']):
        words = nltk.word_tokenize(question['question'])
        question_token = []

        for word in words:
            try:
                question_token.append(word_dic[word])

            except:
                question_token.append(word_index)
                word_dic[word] = word_index
                word_index += 1

        answer_word = question['answer']

        try:
            answer = answer_dic[answer_word]

        except:
            answer = answer_index
            answer_dic[answer_word] = answer_index
            answer_index += 1

        result.append((question['image_filename'], question_token, answer,
                    question['question_family_index']))

    with open(os.path.join(out_dir, f'{split}.pkl'), 'wb') as f:
        pickle.dump(result, f)

    return word_dic, answer_dic

if __name__ == '__main__':
    root = sys.argv[1]
    preprocessed_dir = os.path.join(root, 'preprocessed')

    if not os.path.exists(preprocessed_dir):
        os.mkdir(preprocessed_dir)

    word_dic, answer_dic = process_question(root, preprocessed_dir, 'train')
    process_question(root, preprocessed_dir, 'val', word_dic, answer_dic)

    with open(os.path.join(preprocessed_dir, 'dic.pkl'), 'wb') as f:
        pickle.dump({'word_dic': word_dic, 'answer_dic': answer_dic}, f)