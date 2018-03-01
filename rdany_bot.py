import pandas as pd
import numpy as np
import re 
from keras.layers import Input, Embedding, LSTM, Dense, RepeatVector, Bidirectional, Dropout
#from keras.optimizers import Adam, SGD
from keras.models import Model
from keras.models import Sequential
from keras.layers import Activation, Dense
from keras.callbacks import EarlyStopping
from keras.preprocessing import sequence
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
#from keras.preprocessing.sequence import pad_sequences
from keras.callbacks import ModelCheckpoint
from collections import Counter
import nltk


import pandas as pd
df = pd.read_csv('rdany_conversations_2016-03-01.csv')

BATCH_SIZE = 64
GLOVE_EMBEDDING_SIZE = 100
HIDDEN_UNITS = 512
MAX_INPUT_SEQ_LENGTH = 40
MAX_TARGET_SEQ_LENGTH = 40
MAX_VOCAB_SIZE = 10000
WEIGHT_FILE_PATH = 'model1.h5'
WHITELIST = 'abcdefghijklmnopqrstuvwxyz1234567890?.,'
epochs = 100   


human_lines = open('human_text.txt', encoding='utf-8', errors='ignore').read().split('\n')
robot_lines = open('robot_text.txt', encoding='utf-8', errors='ignore').read().split('\n')

questions = human_lines
answers = robot_lines  

def clean_text(text):
    text = text.lower()    
    text = re.sub(r"i'm", "i am", text)
    text = re.sub(r"he's", "he is", text)
    text = re.sub(r"she's", "she is", text)
    text = re.sub(r"it's", "it is", text)
    text = re.sub(r"that's", "that is", text)
    text = re.sub(r"what's", "that is", text)
    text = re.sub(r"where's", "where is", text)
    text = re.sub(r"how's", "how is", text)
    text = re.sub(r"\'ll", " will", text)
    text = re.sub(r"\'ve", " have", text)
    text = re.sub(r"\'re", " are", text)
    text = re.sub(r"\'d", " would", text)
    text = re.sub(r"\'re", " are", text)
    text = re.sub(r"won't", "will not", text)
    text = re.sub(r"can't", "cannot", text)
    text = re.sub(r"n't", " not", text)
    text = re.sub(r"n'", "ng", text)
    text = re.sub(r"'bout", "about", text)
    text = re.sub(r"'til", "until", text)
    text = re.sub(r"[-()\"#/@;:<>{}`+=~|.!?,]", "", text)
    
    return text

clean_questions = []
for question in questions:
    clean_questions.append(clean_text(question))
    
clean_answers = []    
for answer in answers:
        
    clean_answers.append(clean_text(answer))

lengths = []
for question in clean_questions:
    lengths.append(len(question.split()))
for answer in clean_answers:
    lengths.append(len(answer.split()))

lengths = pd.DataFrame(lengths, columns=['counts'])

print(lengths.describe)

word2em = dict()
f = open('glove.6B.100d.txt',encoding="utf8")
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    word2em[word] = coefs
f.close()
print('Loaded %s word vectors.' % len(word2em))

target_counter = Counter()
input_counter = Counter()

ques=[]
for sent in clean_questions:
    words = []
    words = sent.split()
    for w in words:
            input_counter[w] += 1
    ques.append(words)
ans=[]
for sent in clean_answers:
    words = []
    words = sent.split()
    words.insert(0, 'start')
    words.append('end')        
    for w in words:
            target_counter[w] += 1
    ans.append(words)

target_word2idx = dict()
for idx, word in enumerate(target_counter.most_common(70000)):
    target_word2idx[word[0]] = idx + 1

if 'unknown' not in target_word2idx:
    target_word2idx['unknown'] = 0

input_word2idx = dict()
for idx, word in enumerate(input_counter.most_common(70000)):
    input_word2idx[word[0]] = idx + 1

if 'unknown' not in input_word2idx:
    input_word2idx['unknown'] = 0

target_idx2word = dict([(idx, word) for word, idx in target_word2idx.items()])
input_idx2word = dict([(idx, word) for word, idx in input_word2idx.items()])

num_decoder_tokens = len(target_idx2word)+1
num_encoder_tokens = len(input_idx2word)+1

input_texts_word2em = []

encoder_max_seq_length = 0
decoder_max_seq_length = 0

for input_words, target_words in zip(ques, ans):
    encoder_input_wids = []
    for w in input_words:
        emb = np.zeros(shape=100)
        if w in word2em:
            emb = word2em[w]
        encoder_input_wids.append(emb)

    input_texts_word2em.append(encoder_input_wids)
    encoder_max_seq_length = max(len(encoder_input_wids), encoder_max_seq_length)
    decoder_max_seq_length = max(len(target_words), decoder_max_seq_length)

context = dict()
context['num_decoder_tokens'] = num_decoder_tokens
context['encoder_max_seq_length'] = encoder_max_seq_length
context['decoder_max_seq_length'] = decoder_max_seq_length
context['num_encoder_tokens'] = num_encoder_tokens

print(context)

def generate_batch(input_word2em_data, output_text_data):
    num_batches = len(input_word2em_data) // BATCH_SIZE
    while True:
        for batchIdx in range(0, num_batches):
            start = batchIdx * BATCH_SIZE
            end = (batchIdx + 1) * BATCH_SIZE
            encoder_input_data_batch = np.array(pad_sequences(input_word2em_data[start:end], encoder_max_seq_length))
            decoder_target_data_batch = np.zeros(shape=(BATCH_SIZE, decoder_max_seq_length, num_decoder_tokens))
            decoder_input_data_batch = np.zeros(shape=(BATCH_SIZE, decoder_max_seq_length, GLOVE_EMBEDDING_SIZE))
            for lineIdx, target_words in enumerate(output_text_data[start:end]):
                for idx, w in enumerate(target_words):
                    w2idx = target_word2idx['unknown']  
                    if w in target_word2idx:
                        w2idx = target_word2idx[w]
                    if w in word2em:
                        decoder_input_data_batch[lineIdx, idx, :] = word2em[w]
                    if idx > 0:
                        decoder_target_data_batch[lineIdx, idx - 1, w2idx] = 1
            yield [encoder_input_data_batch, decoder_input_data_batch], decoder_target_data_batch


encoder_inputs = Input(shape=(None, GLOVE_EMBEDDING_SIZE), name='encoder_inputs')
encoder_lstm = LSTM(units=HIDDEN_UNITS, return_state=True, name='encoder_lstm')

encoder_outputs, encoder_state_h, encoder_state_c = encoder_lstm(encoder_inputs)
encoder_states = [encoder_state_h, encoder_state_c]

decoder_inputs = Input(shape=(None, GLOVE_EMBEDDING_SIZE), name='decoder_inputs')
decoder_lstm = LSTM(units=HIDDEN_UNITS, return_state=True, return_sequences=True, name='decoder_lstm')
decoder_outputs, decoder_state_h, decoder_state_c = decoder_lstm(decoder_inputs,
                                                                 initial_state=encoder_states)
decoder_dense = Dense(units=num_decoder_tokens, activation='softmax', name='decoder_dense')
decoder_outputs = decoder_dense(decoder_outputs)

model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

Xtrain, Xtest, Ytrain, Ytest = train_test_split(input_texts_word2em, ans, test_size=0.2, random_state=42)

print(len(Xtrain))
print(len(Xtest))

train_gen = generate_batch(Xtrain, Ytrain)
test_gen = generate_batch(Xtest, Ytest)

train_num_batches = len(Xtrain)// BATCH_SIZE 
test_num_batches = len(Xtest) // BATCH_SIZE

checkpoint = ModelCheckpoint(filepath=WEIGHT_FILE_PATH, save_best_only=True)

model.fit_generator(generator=train_gen, steps_per_epoch=train_num_batches,
                    epochs=100,
                    verbose=1, validation_data=test_gen,
                    validation_steps=test_num_batches,
                    callbacks=[checkpoint])

model.save_weights(WEIGHT_FILE_PATH)
#model.load_weights('model1.h5')

#model.compile(optimizer='rmsprop', loss='categorical_crossentropy')

encoder_model = Model(encoder_inputs, encoder_states)

decoder_state_inputs = [Input(shape=(HIDDEN_UNITS,)), Input(shape=(HIDDEN_UNITS,))]
decoder_outputs, state_h, state_c = decoder_lstm(decoder_inputs, initial_state=decoder_state_inputs)
decoder_states = [state_h, state_c]
decoder_outputs = decoder_dense(decoder_outputs)

decoder_model = Model([decoder_inputs] + decoder_state_inputs, [decoder_outputs] + decoder_states)

def in_white_list(_word):
    for char in _word:
        if char in WHITELIST:
            return True
    return False

def reply(input_text):
    input_seq = []
    input_emb = []
    for word in input_text.lower().split():
        if not in_white_list(word):
            continue
        emb = np.zeros(shape=GLOVE_EMBEDDING_SIZE)
        if word in word2em:
            emb = word2em[word]
        input_emb.append(emb)
    input_seq.append(input_emb)
    input_seq = pad_sequences(input_seq,MAX_INPUT_SEQ_LENGTH)
    states_value = encoder_model.predict(input_seq)
    target_seq = np.zeros((1, 1, GLOVE_EMBEDDING_SIZE))
    target_seq[0, 0, :] = word2em['start']
    target_text = ''
    target_text_len = 0
    terminated = False
    while not terminated:
        output_tokens, h, c = decoder_model.predict([target_seq] + states_value)

        sample_token_idx = np.argmax(output_tokens[0, -1, :])
        sample_word = target_idx2word[sample_token_idx]
        target_text_len += 1

        if sample_word != 'start' and sample_word != 'end':
            target_text += ' ' + sample_word

        if sample_word == '\n' or target_text_len >= MAX_TARGET_SEQ_LENGTH or sample_word == 'end':
            terminated = True

        target_seq = np.zeros((1, 1, GLOVE_EMBEDDING_SIZE))
        if sample_word in word2em:
            target_seq[0, 0, :] = word2em[sample_word]

        states_value = [h, c]
    return target_text.strip()

def test_run():
    print(reply('what is IMDB rating of captian america?')) 
    print(reply('then go fot it'))
    print(reply('what is wikipeda'))
test_run()

