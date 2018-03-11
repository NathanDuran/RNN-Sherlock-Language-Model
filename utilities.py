import numpy as np
import pickle


def save_training_data(path, data):
    file = open(path, "wb")
    pickle.dump(data, file, protocol=pickle.HIGHEST_PROTOCOL)
    file.close()
    print("Saved file training data to %s." % path)


def load_training_data(path):
    with open(path, 'rb') as file:
        saved_data = pickle.load(file)
        file.close()
    print("Loaded file training data from %s." % path)
    return saved_data


def save_model_parameters(path, model, model_type):
    if model_type is "rnn":
        np.savez(path, input_weights=model.input_weights, output_weights=model.output_weights, hidden_weights=model.hidden_weights)
    elif model_type is "lstm":
        np.savez(path, input_weights_g=model.input_weights_g, input_weights_i=model.input_weights_i, input_weights_f=model.input_weights_f, input_weights_o=model.input_weights_o,
                 hidden_weights_g=model.hidden_weights_g, hidden_weights_i=model.hidden_weights_i, hidden_weights_f=model.hidden_weights_f, hidden_weights_o=model.hidden_weights_o,
                 bias_g=model.bias_g, bias_i=model.bias_i, bias_f=model.bias_f, bias_o=model.bias_o,
                 output_weights=model.output_weights, bias_output=model.bias_output)
    print("Saved model parameters to %s." % path)


def load_model_parameters(path, model, model_type):
    npzfile = np.load(path)
    if model_type is "rnn":
        model.input_weights = npzfile["input_weights"]
        model.output_weights = npzfile["output_weights"]
        model.hidden_weights = npzfile["hidden_weights"]
        model.hidden_dimension = model.input_weights.shape[0]
        model.word_dimension = model.input_weights.shape[1]
    elif model_type is "lstm":
        model.input_weights_g, model.input_weights_i, model.input_weights_f, model.input_weights_o = npzfile["input_weights_g"], npzfile["input_weights_i"], npzfile["input_weights_f"], npzfile["input_weights_o"]
        model.hidden_weights_g, model.hidden_weights_i, model.hidden_weights_f, model.hidden_weights_o = npzfile["hidden_weights_g"], npzfile["hidden_weights_i"], npzfile["hidden_weights_f"], npzfile["hidden_weights_o"]
        model.bias_g, model.bias_i, model.bias_f, model.bias_o = npzfile["bias_g"], npzfile["bias_i"], npzfile["bias_f"], npzfile["bias_o"]
        model.output_weights, model.bias_output = npzfile["output_weights"], npzfile["bias_output"]
    print("Loaded model parameters from %s. " % path)
    return model
