# inspi to function comes from:
# https://towardsdatascience.com/visualize-bert-sequence-embeddings-an-unseen-way-1d6a351e4568


from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns 
import numpy as np

def visualize_layerwise_embeddings(hidden_states,labels,title,layers_to_visualize, perplexity, init):
    dim_reducer = TSNE(n_components=2, init = init, perplexity = perplexity)

    num_layers = len(layers_to_visualize)
    
    col_size = int(num_layers/2)
    print('>>> col size:',col_size)
    fig = plt.figure(figsize=(24,col_size*6)) #each subplot of size 6x6, each row will hold 4 plots
    ax = [fig.add_subplot(col_size,2,i+1) for i in range(num_layers)]
    
    labels = np.array(labels).reshape(-1)
    for i,layer_i in enumerate(layers_to_visualize):
        layer_embeds = hidden_states[layer_i]
        
        layer_averaged_hidden_states = layer_embeds.sum(axis=1)
        layer_dim_reduced_embeds = dim_reducer.fit_transform(layer_averaged_hidden_states);
        
        df = pd.DataFrame.from_dict({'x':layer_dim_reduced_embeds[:,0],'y':layer_dim_reduced_embeds[:,1],'label':labels})
        
        sns.scatterplot(data=df,x='x',y='y',hue='label',ax=ax[i]).set(title = f'Layer: {layer_i}');
        
    plt.savefig(f'/work3/s174498/roberta_files/hidden_layers_{title}_{perplexity}_{init}',format='png',pad_inches=0)