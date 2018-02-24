import numpy as np
import pickle

def softmax(x):
    xt = np.exp(x - np.max(x))
    return xt / np.sum(xt)


def save_training_data(path, data):
    save_data = dict(
        x_train=data.x_train,
        y_train=data.y_train,
        word_to_index=data.word_to_index,
        index_to_word=data.index_to_word,)

    file = open(path, "wb")
    pickle.dump(save_data, file, protocol=pickle.HIGHEST_PROTOCOL)
    file.close()
    print("Saved file training data to %s." % path)


def load_training_data(path, data):
    with open(path, 'rb') as file:
        saved_data = pickle.load(file)
        x_train, y_train = saved_data["x_train"], saved_data["y_train"]
        word_to_index, index_to_word = saved_data["word_to_index"], saved_data["index_to_word"]
        data.x_train = x_train
        data.y_train = y_train
        data.word_to_index = word_to_index
        data.index_to_word = index_to_word
        file.close()
    print("Loaded file training data from %s." % path)
    return data


def save_model_parameters(path, model):
    input_weights, output_weights, hidden_weights = model.input_weights, model.output_weights, model.hidden_weights
    np.savez(path, U=input_weights, V=output_weights, W=hidden_weights)
    print("Saved model parameters to %s." % path)


def load_model_parameters(path, model):
    npzfile = np.load(path)
    U, V, W = npzfile["U"], npzfile["V"], npzfile["W"]
    model.hidden_dimension = U.shape[0]
    model.word_dimemsion = U.shape[1]
    model.input_weights = U
    model.output_weights = V
    model.hidden_weights = W
    print("Loaded model parameters from %s. hidden_dimension=%d word_dimension=%d" % (path, U.shape[0], U.shape[1]))
    return model