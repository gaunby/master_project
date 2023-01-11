from scipy.stats import ttest_ind, ttest_1samp
import numpy as np
import matplotlib.pyplot as plt

import seaborn as sns 
import pandas as pd


def plot_results(results, target , plot_concepts,PATH,plot_hist = False, save_fig = False, bonferroni_nr = None,min_p_val=0.05,
    t_test_mean = None):
  """Helper function to organize results.
  When run in a notebook, outputs a matplotlib bar plot of the
  TCAV scores for all bottlenecks for each concept, replacing the
  bars with asterisks when the TCAV score is not statistically significant.
  Args:
    results: dictionary of results from TCAV runs.
    target: str of negative or positive as target class
    plot_concept: list of concepts in resutls

    random_counterpart: name of the random_counterpart used, if it was used. 
    random_concepts: list of random experiments that were run. 
    num_random_exp: number of random experiments that were run.
    min_p_val: minimum p value for statistical significance
    t_test_mean: if value is given a 1-sample t-test will beformed
    plot_hist: to plot histograms of results 
    alternative: input to t-test 
  """

  if bonferroni_nr == None:
    bonferroni_nr = 1
  print(PATH)
  min_p_val = min_p_val/bonferroni_nr
    
  # to plot, must massage data again 
  plot_data = {}
  plot_concepts 

  df_result = pd.DataFrame(columns = ['TCAV score','Concept','Bottleneck'])
  # print concepts and classes with indentation
  for concept in plot_concepts:#results[target]:
    print('concept', concept)
    result = results[target][concept]
    
    for bottleneck in result:      
      if bottleneck not in df_result['Bottleneck'].unique():
        df_random = pd.DataFrame(results[target]['random'][bottleneck]['TCAV'],columns=['TCAV score'])
        df_random['Concept'] = 'random'
        df_random['Bottleneck'] = bottleneck
        df_result = pd.concat([df_result, df_random], ignore_index=True)

      df_concept = pd.DataFrame( result[bottleneck]['TCAV'],columns=['TCAV score'])
      df_concept['Concept'] = concept
      df_concept['Bottleneck'] = bottleneck
      df_result = pd.concat([df_result, df_concept], ignore_index=True)

      # Calculate statistical significance
      if t_test_mean == None:
          _, p_val = ttest_ind(results[target]['random'][bottleneck]['TCAV'], result[bottleneck]['TCAV'])

      else:
          _, p_val = ttest_1samp(result[bottleneck]['TCAV'], t_test_mean)
          _, p_val_random = ttest_1samp(results[target]['random'][bottleneck]['TCAV'], t_test_mean)
      
      if bottleneck not in plot_data:
        plot_data[bottleneck] = {'random_p-value':[], 'bn_vals': [], 'bn_stds': [], 'significant': [], 'p-value': [], 'concept':[]}
        plot_data[bottleneck]['random_p-value'].append(np.mean(results[target]['random'][bottleneck]['TCAV']))
        plot_data[bottleneck]['random_p-value'].append(np.std(results[target]['random'][bottleneck]['TCAV']))
        if t_test_mean != None:  
          plot_data[bottleneck]['random_p-value'].append(p_val_random)
          if p_val_random > min_p_val:
              plot_data[bottleneck]['random_p-value'].append('False')
          else:
              plot_data[bottleneck]['random_p-value'].append('True')
      
      if p_val > min_p_val:
        # statistically insignificant
        plot_data[bottleneck]['bn_vals'].append(0.01)
        plot_data[bottleneck]['bn_stds'].append(0)
        plot_data[bottleneck]['significant'].append(False)
        plot_data[bottleneck]['p-value'].append(p_val)
        plot_data[bottleneck]['concept'].append(concept)
          
      else:
        plot_data[bottleneck]['bn_vals'].append(np.mean(result[bottleneck]['TCAV']))
        plot_data[bottleneck]['bn_stds'].append(np.std(result[bottleneck]['TCAV']))
        plot_data[bottleneck]['significant'].append(True)
        plot_data[bottleneck]['p-value'].append(p_val)
        plot_data[bottleneck]['concept'].append(concept)

  # histogram plots
  if plot_hist:

    palette ={ "random": "grey",
    "news": "darkblue", "world": "darkorange",'sport':'g','business':'darkviolet','science':'darkred', 
    "woman": "darkblue", "transsexual": "darkorange", "intersex": "g", "man": "darkviolet",
    'hate': "darkblue",'irony': "darkorange",'offensive':'g' ,
    "gender": "darkorange",}
    i = 0
    
    nr_plots = len(plot_concepts)
    for nr, bottlenecks in enumerate(df_result['Bottleneck'].unique()):
      data = df_result[df_result['Bottleneck'] == bottlenecks]
      # set figure
      plt.subplots(nrows=1, ncols=nr_plots, sharey=True,figsize=(15,4));
      plt.suptitle(f'Histogram of TCAV scores for Concepts in Layer {nr}', fontsize =20);
      for hist_nr in range(len(plot_concepts)):
        plt.subplot(1, nr_plots, hist_nr+1);
        ax = sns.histplot(data=data[data['Concept'].isin([plot_concepts[hist_nr],'random'])], x="TCAV score", hue_order =[plot_concepts[hist_nr],'random'],
        hue="Concept", stat = 'percent', binrange = (0,1),common_norm=False, bins = 20, element="step", palette=palette);
        sns.move_legend( ax, loc = "upper left", fontsize = 'x-large');
        ax.set_xlabel("TCAV score",fontsize = 'xx-large');
        ax.set_ylabel("Percent",fontsize =  'xx-large');
        plt.axvline(0.5, 0,10, ls = '--', lw = 0.8, color = 'grey');
      # finish figure
      plt.tight_layout();
      if save_fig:
        print('Now overwritting and saving figure')
        fig_path = PATH + 'histogram_'+target+'_'+''.join(plot_concepts)+'_'+str(i)+'.pdf' 
        i += 1
        plt.savefig(fig_path)
      plt.show();
  
  
      
  # subtract number of random experiments
  print(plot_data[bottleneck])
  num_concepts = len(plot_concepts) 
  print('num concepts', num_concepts)
  num_bottlenecks = len(plot_data)
  bar_width = 0.35
    
  # create location for each bar. scale by an appropriate factor to ensure 
  # the final plot doesn't have any parts overlapping
  index = np.arange(num_concepts) * bar_width * (num_bottlenecks + 1)

  fig, ax = plt.subplots(figsize = (15,6))
  # draw all bottlenecks individually
  for i, [bn, vals] in enumerate(plot_data.items()):
    print('bn',bn)
    label_text = f'{i}' 
    bar = ax.bar(index + i * bar_width, vals['bn_vals'],
        bar_width, yerr=vals['bn_stds'],ecolor = 'grey', label=label_text, color = sns.color_palette("Paired")[i])
    # draw stars to mark bars that are stastically insignificant to 
    # show them as different from others
    for j, significant in enumerate(vals['significant']):
      if not significant:
        ax.text(index[j] + i * bar_width - 0.1, 0.01, "*",
            fontdict = {'weight': 'bold', 'size': 16,
            'color': bar.patches[0].get_facecolor()})
  # set properties
  ax.set_title('TCAV scores for Concepts and Layers', fontsize = 20)
  ax.set_ylabel('TCAV score', fontsize = 'xx-large')
  ax.set_xticks(index + num_bottlenecks * bar_width / 2)
  
  plot_concepts = ['gender','gender:intersex','gender:man','gender:woman']
  
  ax.set_xticklabels(plot_concepts, fontsize = 'xx-large')#fontsize = 16)
  ax.legend(fontsize = 'large',loc='center left', bbox_to_anchor=(1, 0.5), title = 'Layer')#, ('layer 0','layer 1','layer 2','layer 3','layer 4','layer 5','layer 6','layer 7','layer 8','layer 9','layer 10','layer 11'))
  fig.tight_layout()
  if save_fig:
    i = 0 
    print('Now overwritting and saving figure')
    fig_path = PATH +'barplot_'+target+'_'+''.join(plot_concepts)+'_bonferroni_'+str(bonferroni_nr)+'.pdf'
    plt.savefig(fig_path)

  return plot_data



def plot_results_mark(results, target , plot_concepts,PATH,plot_hist = False, save_fig = False, bonferroni_nr = None,min_p_val=0.05,
    t_test_mean = None):

  if bonferroni_nr == None:
    bonferroni_nr = 1
  print(PATH)
  min_p_val = min_p_val/bonferroni_nr

  # to plot, must massage data again 
  plot_data = {}
  plot_concepts 

  df_result = pd.DataFrame(columns = ['TCAV score','Concept','Bottleneck'])
  # print concepts and classes with indentation
  for concept in plot_concepts:#results[target]:
    print('concept', concept)
    result = results[target][concept]
    
    for bottleneck in result:      
      if bottleneck not in df_result['Bottleneck'].unique():
        df_random = pd.DataFrame(results[target]['random'][bottleneck]['TCAV'],columns=['TCAV score'])
        df_random['Concept'] = 'random'
        df_random['Bottleneck'] = bottleneck
        df_result = pd.concat([df_result, df_random], ignore_index=True)

      df_concept = pd.DataFrame( result[bottleneck]['TCAV'],columns=['TCAV score'])
      df_concept['Concept'] = concept
      df_concept['Bottleneck'] = bottleneck
      df_result = pd.concat([df_result, df_concept], ignore_index=True)

      # Calculate statistical significance
      if t_test_mean == None:
          _, p_val = ttest_ind(results[target]['random'][bottleneck]['TCAV'], result[bottleneck]['TCAV'])

      else:
          _, p_val = ttest_1samp(result[bottleneck]['TCAV'], t_test_mean)
          _, p_val_random = ttest_1samp(results[target]['random'][bottleneck]['TCAV'], t_test_mean)
      
      if bottleneck not in plot_data:
        plot_data[bottleneck] = {'random_p-value':[], 'bn_vals': [], 'bn_stds': [], 'significant': [], 'p-value': [], 'concept':[]}
        plot_data[bottleneck]['random_p-value'].append(np.mean(results[target]['random'][bottleneck]['TCAV']))
        plot_data[bottleneck]['random_p-value'].append(np.std(results[target]['random'][bottleneck]['TCAV']))
        if t_test_mean != None:  
          plot_data[bottleneck]['random_p-value'].append(p_val_random)
          if p_val_random > min_p_val:
              plot_data[bottleneck]['random_p-value'].append('False')
          else:
              plot_data[bottleneck]['random_p-value'].append('True')
      
      if p_val > min_p_val:
        A = 8
        # statistically insignificant
        plot_data[bottleneck]['bn_vals'].append(0)
        plot_data[bottleneck]['bn_stds'].append(0)
        plot_data[bottleneck]['significant'].append(False)
        plot_data[bottleneck]['p-value'].append(p_val)
        plot_data[bottleneck]['concept'].append(concept)
          
      else:
        plot_data[bottleneck]['bn_vals'].append(np.mean(result[bottleneck]['TCAV']))
        plot_data[bottleneck]['bn_stds'].append(np.std(result[bottleneck]['TCAV']))
        plot_data[bottleneck]['significant'].append(True)
        plot_data[bottleneck]['p-value'].append(p_val)
        plot_data[bottleneck]['concept'].append(concept)

  # histogram plots
  if plot_hist:

    palette ={ "random": "grey",
    "news": "darkblue", "world": "darkorange",'sport':'g','business':'darkviolet','science':'darkred', 
    "woman": "darkblue", "transsexual": "darkorange", "intersex": "g", "man": "darkviolet",
    'hate': "darkblue",'irony': "darkorange",'offensive':'g' ,
    "gender": "darkorange",}
    i = 0
    
    nr_plots = len(plot_concepts)
    for nr, bottlenecks in enumerate(df_result['Bottleneck'].unique()):
      data = df_result[df_result['Bottleneck'] == bottlenecks]
      # set figure
      plt.subplots(nrows=1, ncols=nr_plots, sharey=True,figsize=(15,4));
      plt.suptitle(f'Histogram of TCAV scores for Concepts in Layer {nr}', fontsize =20);
      for hist_nr in range(len(plot_concepts)):
      # first concept
        plt.subplot(1, nr_plots, hist_nr+1);
        ax = sns.histplot(data=data[data['Concept'].isin([plot_concepts[hist_nr],'random'])], x="TCAV score", hue_order =[plot_concepts[hist_nr],'random'],
        hue="Concept", stat = 'percent', binrange = (0,1),common_norm=False, bins = 20, element="step", palette=palette);
        sns.move_legend( ax, loc = "upper left", fontsize = 'x-large');
        ax.set_xlabel("TCAV score",fontsize = 'xx-large');
        ax.set_ylabel("Percent",fontsize =  'xx-large');
        plt.axvline(0.5, 0,10, ls = '--', lw = 0.8, color = 'grey');
      # finish figure
      plt.tight_layout();
      if save_fig:
        print('Now overwritting and saving figure')
        fig_path = PATH + 'histogram_'+target+'_'+''.join(plot_concepts)+'_'+str(i)+'.pdf' 
        i += 1
        plt.savefig(fig_path)
      plt.show();
  
  
      
  # subtract number of random experiments
  num_concepts = len(plot_concepts)
  print('num concepts', num_concepts)
  num_bottlenecks = len(plot_data)
  bar_width = 0.35
    
  # create location for each bar. scale by an appropriate factor to ensure 
  index = np.arange(num_concepts) * bar_width * (num_bottlenecks + 1)

  # set up bar plot  
  fig, ax = plt.subplots(figsize = (15,6))
  # draw all bottlenecks individually
  for i, [bn, vals] in enumerate(plot_data.items()):
    label_text = f'{i}' 
    bar = ax.bar(index + i * bar_width, vals['bn_vals'],
        bar_width, yerr=vals['bn_stds'],ecolor = 'grey', label=label_text, color = sns.color_palette("Paired")[i])
    # mark bars that are stastically insignificant to show them as different from others
    for j, significant in enumerate(vals['significant']):
      if not significant:
        ax.plot(index[j] + i * bar_width, 0.5,marker =  "x", color=  bar.patches[0].get_facecolor() , markersize = 10)
  # set properties
  ax.set_title('TCAV scores for Concepts and Layers', fontsize = 20)
  ax.set_ylabel('TCAV score', fontsize = 'xx-large')
  ax.set_xticks(index + num_bottlenecks * bar_width / 2)
  
  plot_concepts = ['gender','gender:intersex','gender:man','gender:woman']
  
  ax.set_xticklabels(plot_concepts, fontsize = 'xx-large')
  ax.legend(fontsize = 'large',loc='center left', bbox_to_anchor=(1, 0.5), title = 'Layer')
  plt.axhline(y = 0.5, color = 'lightgrey', linestyle = '--',lw = 0.8)
  fig.tight_layout()
  if save_fig:
    i = 0 
    print('Now overwritting and saving figure')
    fig_path = PATH +'barplot_'+target+'_'+''.join(plot_concepts)+'_bonferroni_'+str(bonferroni_nr)+'.pdf'
    plt.savefig(fig_path)

  return plot_data