import pickle

# PARAMETERS 
file_name = 'negative_layer_3_11' # name of saved file 


# loading tcav data
PATH =  f"/work3/s174498/nlp_tcav_results/{file_name}.pkl"
f = open(PATH, 'rb')
tcav_dict = pickle.load(f)
f.close()

