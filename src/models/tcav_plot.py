import pickle

# PARAMETERS 
file_name = 'negative_layer_3_11' # name of saved file 


# loading tcav data
PATH =  f"/work3/s174498/nlp_tcav_results/{file_name}.pkl"
f = open(PATH, 'rb')
tcav_dict = pickle.load(f)
f.close()

layers = [#'roberta.encoder.layer.0.output.dense',
        #'roberta.encoder.layer.1.output.dense',
        #'roberta.encoder.layer.2.output.dense',
        'roberta.encoder.layer.3.output.dense',
        'roberta.encoder.layer.4.output.dense',
        'roberta.encoder.layer.5.output.dense',
        'roberta.encoder.layer.6.output.dense',
        'roberta.encoder.layer.7.output.dense',
        'roberta.encoder.layer.8.output.dense',
        'roberta.encoder.layer.9.output.dense',
        'roberta.encoder.layer.10.output.dense',
        'roberta.encoder.layer.11.output.dense'
        ]


tcav_dict['negative']['hate'] 