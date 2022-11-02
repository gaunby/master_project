# inspi to function comes from:
# https://towardsdatascience.com/visualize-bert-sequence-embeddings-an-unseen-way-1d6a351e4568


from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns 
import numpy as np

def visualize_layerwise_embeddings(hidden_states,labels,title,layers_to_visualize, perplexity, init, save = False):
    dim_reducer = TSNE(n_components=2, init = init, perplexity = perplexity)

    num_layers = len(layers_to_visualize)
    n = len(np.array(labels).reshape(-1))
    label_text = np.array(['nan']*n)
    label_text = np.where(np.array(labels).reshape(-1) == 0, label_text, 'neg')
    label_text = np.where(np.array(labels).reshape(-1) == 1, label_text, 'pos')

    col_size = int(num_layers/2)
    fig = plt.figure(figsize=(15,col_size*4), constrained_layout=True) #each subplot of size 6x6, each row will hold 4 plots
    fig.suptitle(f'T-SNE results of hidden layers in RoBERTa', fontsize =24)
    ax = [fig.add_subplot(col_size,2,i+1) for i in range(num_layers)]
    
    palette ={0:sns.color_palette("Paired")[0], 1:sns.color_palette("Paired")[6],
                'neg':sns.color_palette("Paired")[0], 'pos':sns.color_palette("Paired")[6]}

    labels = np.array(labels).reshape(-1)

    for i,layer_i in enumerate(layers_to_visualize):
        layer_embeds = hidden_states[layer_i]
        
        layer_averaged_hidden_states = layer_embeds.mean(axis=1)
        layer_dim_reduced_embeds = dim_reducer.fit_transform(layer_averaged_hidden_states);
        
        df = pd.DataFrame.from_dict({'first dim':layer_dim_reduced_embeds[:,0],'second dim':layer_dim_reduced_embeds[:,1],'label':labels, 'label_text': label_text})
        
        scatter = sns.scatterplot(data=df,x='first dim',y='second dim',hue='label_text',ax=ax[i], palette = palette, alpha = 0.8 )
        scatter.legend(loc = 'upper right',fontsize = 'x-large', title = 'class',title_fontsize='x-large');
        scatter.axes.set_title( f'Layer {layer_i}',fontsize='xx-large')
        scatter.set_xlabel("x",fontsize='xx-large');
        scatter.set_ylabel("y",fontsize='xx-large');
        scatter.tick_params(labelsize='x-large')

    #fig.tight_layout();
    if save:    
        plt.savefig(f'/work3/s174498/roberta_files/hidden_layers_{title}_{perplexity}_{init}',format='png',pad_inches=0)



def visualize_one_layer(hidden_state,labels,title, perplexity, init, save = False):
    dim_reducer = TSNE(n_components=2, init = init, perplexity = perplexity)
    
    plt.figure(figsize=(10,6)) #each subplot of size 6x6, each row will hold 4 plots
    plt.title(f't-SNE of embedding in RoBERTa')

    labels = np.array(labels).reshape(-1)
    
    layer_averaged_hidden_states = hidden_state.mean(axis=1)
    layer_dim_reduced_embeds = dim_reducer.fit_transform(layer_averaged_hidden_states);
    print('>>> t-SNE output dim:',layer_dim_reduced_embeds.shape)
    df = pd.DataFrame.from_dict({'x':layer_dim_reduced_embeds[:,0],'y':layer_dim_reduced_embeds[:,1],'label':labels})
    
    scatter = sns.scatterplot(data=df,x='x',y='y',hue='label');
    sns.move_legend( scatter, loc = "upper left", fontsize = 'x-large');
    scatter.set_xlabel(fontsize = 'xx-large');
    scatter.set_ylabel(fontsize =  'xx-large');

    if save:    
        plt.savefig(f'/work3/s174498/roberta_files/hidden_layers_{title}_{perplexity}_{init}',format='png',pad_inches=0)