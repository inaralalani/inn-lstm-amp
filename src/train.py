#!/usr/bin/env python
'''
amp_train_and_predict_cnn-lstm_model.py By: Dan Veltri
Takes a protein multi-FASTA file as input and outputs prediction values for each.
Assumes peptides are >= 10 and <= 200 AA in length (AA's longer than 'max_length' are ignored).

Prediction probabilities >= 0.5 predict AMPs and < 0.5 non-AMPs.

User should provide training, validation, and testing FASTA files for AMPs and decoys, respectivley. 
Ensure 'max_length' is >= to the longest peptide in those files.

NOTE on a multi-threaded machine TF version 1.x will still produce stochastic results even if you set a random seed.

While best efforts have been made to ensure the integrity of this script, we take no
responsibility for damages that may result from its use.
'''
# from __future__ import print_function # enable Python3 printing
from pprint import pprint
import os
import numpy
from sklearn.metrics import confusion_matrix, roc_auc_score, matthews_corrcoef, classification_report
from sklearn.utils import shuffle
from Bio import SeqIO
import tensorflow as tf
import sys

from tensorflow.python.keras.models import Sequential, load_model
from tensorflow.python.keras.layers import Dense, LSTM, Reshape
from tensorflow.python.keras.layers.convolutional import Conv1D, MaxPooling1D
from tensorflow.python.keras.layers.embeddings import Embedding
from tensorflow.keras.preprocessing.sequence import pad_sequences
# from keras.preprocessing import sequence

from layers import Involution1D, Involution2D

assert len(sys.argv) == 2, "must pass cnn or inn as argument"

net_type = sys.argv[1].lower()
assert net_type in ('cnn', 'inn')

# User-set variables
saved_model_name = f'models/amp_model_{net_type}.h5'
amp_train_fasta = 'input/AMP.tr.fa'
amp_validate_fasta = 'input/AMP.eval.fa'
amp_test_fasta = 'input/AMP.te.fa'
decoy_train_fasta = 'input/DECOY.tr.fa'
decoy_validate_fasta = 'input/DECOY.eval.fa'
decoy_test_fasta = 'input/DECOY.te.fa'

# Model params
max_length = 200
embedding_vector_length = 128
nbf = 64 		# No. Conv Filters
flen = 16 		# Conv Filter length
nlstm = 100 	# No. LSTM layers
ndrop = 0.1     # LSTM layer dropout
nbatch = 32 	# Fit batch No.
nepochs = 10    # No. training rounds

amino_acids = "XACDEFGHIKLMNPQRSTVWY"
aa2int = dict((c, i) for i, c in enumerate(amino_acids))

X_train = []
y_train = []
X_val = []
y_val = []
X_test = []
y_test = []

print("Encoding training/testing sequences...")
for s in SeqIO.parse(amp_train_fasta, "fasta"):
    X_train.append([aa2int[aa] for aa in str(s.seq).upper()])
    y_train.append(1)
for s in SeqIO.parse(amp_validate_fasta, "fasta"):
    X_val.append([aa2int[aa] for aa in str(s.seq).upper()])
    y_val.append(1)
for s in SeqIO.parse(amp_test_fasta, "fasta"):
    X_test.append([aa2int[aa] for aa in str(s.seq).upper()])
    y_test.append(1)
for s in SeqIO.parse(decoy_train_fasta, "fasta"):
    X_train.append([aa2int[aa] for aa in str(s.seq).upper()])
    y_train.append(0)
for s in SeqIO.parse(decoy_validate_fasta, "fasta"):
    X_val.append([aa2int[aa] for aa in str(s.seq).upper()])
    y_val.append(0)
for s in SeqIO.parse(decoy_test_fasta, "fasta"):
    X_test.append([aa2int[aa] for aa in str(s.seq).upper()])
    y_test.append(0)

# Pad input sequences
X_train = pad_sequences(X_train, maxlen=max_length)
X_val = pad_sequences(X_val, maxlen=max_length)
X_test = pad_sequences(X_test, maxlen=max_length)

# Shuffle training sequences
X_train, y_train = shuffle(X_train, numpy.array(y_train))
X_val, y_val = shuffle(X_val, numpy.array(y_val))


def compile_model(channel=3, group_number=1, kernel_size=5, stride=1, reduction_ratio=1):
    print("Compiling model...")
    model = Sequential()
    model.add(Embedding(21, embedding_vector_length, input_length=max_length))

    if net_type == 'cnn':
        model.add(Conv1D(filters=nbf, kernel_size=flen,
                  padding="same", activation='relu'))
    if net_type == 'inn':
        model.add(Reshape((200, 128, 1)))
        model.add(Involution2D(channel=channel, group_number=group_number,
                  kernel_size=kernel_size, stride=stride, reduction_ratio=reduction_ratio, name="inv_1"))
        model.add(Reshape((200, 128)))

    model.add(MaxPooling1D(pool_size=5))
    # ,merge_mode='ave'))
    model.add(LSTM(nlstm, use_bias=True, dropout=ndrop, return_sequences=False))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy',
                  optimizer='adam', metrics=['accuracy'])

    return model


model = compile_model()
test_params = {
    'kernel_size': [5, 6, 7, 8, 9, 10]
}

# if os.path.exists('models/weights/checkpoint'):
#     model.load_weights('models/weights/temp.weights')
# else:
#     print("Training now...")
#     model.fit(X_train, numpy.array(y_train), epochs=nepochs, batch_size=nbatch, verbose=1)
#     model.save_weights('models/weights/temp.weights')

# model.save(saved_model_name)
tests = {}


def test_model(model, param, value):

    print("\nGathering Testing Results...")
    preds = model.predict(X_test)
    pred_class = numpy.rint(preds)  # round up or down at 0.5
    true_class = numpy.array(y_test)
    tn, fp, fn, tp = confusion_matrix(true_class, pred_class).ravel()
    roc = roc_auc_score(true_class, preds) * 100.0
    mcc = matthews_corrcoef(true_class, pred_class)
    acc = (tp + tn) / (tn + fp + fn + tp + 0.0) * 100.0
    sens = tp / (tp + fn + 0.0) * 100.0
    spec = tn / (tn + fp + 0.0) * 100.0
    prec = tp / (tp + fp + 0.0) * 100.0

    return {'0_param_value': value, 'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn, 'sens': sens, 'spec': spec, 'acc': acc, 'mcc': mcc, 'auroc': roc, 'prec': prec}


# print("\nTP\tTN\tFP\tFN\tSens\tSpec\tAcc\tMCC\tauROC\tPrec")
# print("{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}".format(tp,tn,fp,fn,numpy.round(sens,4),numpy.round(spec,4),numpy.round(acc,4),numpy.round(mcc,4),numpy.round(roc,4),numpy.round(prec,4)))
# print("\nSaved model as: {}".format(saved_model_name))
test_model(model, 'stride', 1)
pprint(tests)

# END PROGRAM
