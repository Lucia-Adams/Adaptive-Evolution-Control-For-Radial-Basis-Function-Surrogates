import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.pylab as plt

# expected csv format:
# pop_size,n_gen,err_scale,const_sample,nbh_scale,evals,gd_plus,igd_plus,hv,err_plot_string,
# c_evals,gd_plus_norm,igd_plus_norm,hv_norm,graph_counter

def df_table_for_all_err_scalar(csv_filename):
    val_df = pd.read_csv(csv_filename)
    # val_df.err_plot_string = val_df.err_plot_string.str.split('+').apply(lambda list: [float(val) for val in list])
    grouped_df = val_df.groupby(['err_scale','const_sample','nbh_scale']).agg(
        {'evals': 'mean', 'gd_plus':'mean', 'igd_plus':'mean', 'hv':'mean', 'gd_plus_norm':'mean',
        'igd_plus_norm':'mean', 'hv_norm':'mean'})
    # grouped_df = val_df.groupby(['err_scale','const_sample','nbh_scale']).agg(
    #     {'evals': 'mean', 'gd_plus':'mean', 'hv':'mean', 'gd_plus_norm':'mean', 'hv_norm':'mean'})
    round_df = grouped_df.round({'evals':1, 'gd_plus':4, 'hv':4, 'gd_plus_norm':4, 'hv_norm':4})
    return round_df

def scatter_graph_evals_per_gd_plus(problem_name, problem_df, err_scale_list=[None,0.1,0.3,0.5,0.7,0.9]):
    """
    problem_name (str): name of problem for graph
    problem_df (DataFrame): should have err_scale, const_sample and nh_scale grouped by so one entry per combo. Regarding a single problem
    err_scale_list (list): list of all the floats used for error scalars in dataframe. Can include None.
    """
    colours_list = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2',
    '#7f7f7f', '#bcbd22', '#17becf']
    plt.figure(figsize=(5,5.8))

    problem_df['av_rec_gd+_to_evals'] = (1/problem_df["gd_plus"])/problem_df["evals"] 

    row_names = [row.name for _,row in problem_df.iterrows()]
    for i,err in enumerate(err_scale_list):
        # gets dataframe for each error as looping through each error in new colour/set
        row_names_filtered = list(filter(lambda x: x[0]==str(err), row_names))
        e_df = problem_df.loc[row_names_filtered]

        # gets best gd_plus point 
        evals_list = e_df['evals']
        metric_list = list(map(lambda x: 1/x, list(e_df["gd_plus"])))
        best_metric = max(metric_list)
        corresponding_eval = evals_list[metric_list.index(best_metric)]

        # get best gradient point
        gd_over_evals_list = list(e_df['av_rec_gd+_to_evals'])
        best_av = max(gd_over_evals_list)
        best_metric_index = gd_over_evals_list.index(best_av)
        best_av_metric = 1/(list(e_df['gd_plus'])[best_metric_index])
        best_corr_eval = list(e_df['evals'])[best_metric_index]

        middle_metric = (best_metric+best_av_metric)/2
        middle_eval=(corresponding_eval+best_corr_eval)/2

        # plt.plot([0, best_corr_eval], [0, best_av_metric], linestyle='dashed', color=colours_list[i], alpha=0.5) 
        plt.plot([0, middle_eval], [0, middle_metric], linestyle='dashed', color=colours_list[i], alpha=0.6) 
        # Use this one below just for like best metric point
        # plt.plot([0, corresponding_eval], [0, best_metric], linestyle='dashed', color=colours_list[i], alpha=0.5) 
        plt.scatter(evals_list, metric_list, label=f"{err}", color=colours_list[i], s=15)
    
    plt.xlabel('Average Evaluation Count', fontsize=12.5)
    plt.ylabel(f"Average 1/GD Plus", fontsize=12.5)
    plt.legend()  # To show the labels
    plt.tight_layout()
    plt.savefig(f"Scatters/{problem_name}_av_scatter_tall", dpi=400)
    plt.clf()
    plt.close()

def scatter_graph_evals_per_hv(problem_name, problem_df, err_scale_list=[None,0.1,0.3,0.5,0.7,0.9]):
    """
    problem_name (str): name of problem for graph
    problem_df (DataFrame): should have err_scale, const_sample and nh_scale grouped by so one entry per combo. Regarding a single problem
    err_scale_list (list): list of all the floats used for error scalars in dataframe. Can include None.
    """
    colours_list = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2',
    '#7f7f7f', '#bcbd22', '#17becf']
    problem_df['av_hv_to_evals'] = (problem_df["hv"])/problem_df["evals"] 

    row_names = [row.name for _,row in problem_df.iterrows()]
    for i,err in enumerate(err_scale_list):
        # gets dataframe for each error as looping through each error in new colour/set
        row_names_filtered = list(filter(lambda x: x[0]==str(err), row_names))
        e_df = problem_df.loc[row_names_filtered]

        evals_list = e_df['evals']
        metric_list = list(map(lambda x: x, list(e_df["hv"])))
        best_metric = max(metric_list)
        corresponding_eval = evals_list[metric_list.index(best_metric)]

        # get best gradient point
        hv_over_evals_list = list(e_df['av_hv_to_evals'])
        best_av = max(hv_over_evals_list)
        best_metric_index = hv_over_evals_list.index(best_av)
        best_av_metric = (list(e_df['hv'])[best_metric_index])
        best_corr_eval = list(e_df['evals'])[best_metric_index]

        middle_metric = (best_metric+best_av_metric)/2
        middle_eval=(corresponding_eval+best_corr_eval)/2
 
        plt.plot([0, middle_eval], [0, middle_metric], linestyle='dashed', color=colours_list[i], alpha=0.5) 
        # plt.plot([0, corresponding_eval], [0, best_metric], linestyle='dashed', color=colours_list[i], alpha=0.5) 
        plt.scatter(evals_list, metric_list, label=f"{err}", color=colours_list[i], s=10)
  
    plt.xlabel('Average Evaluation Count')
    plt.ylabel(f"Average Hypervolume")
    plt.legend()  # To show the labels
    plt.savefig(f"Scatters/{problem_name}_av_scatter_hv", dpi=400)
    plt.clf()
    plt.close()

def max_evals_per_metric_params(problem_name, problem_df, err_scale_list=[None,0.1,0.3,0.5,0.7,0.9]):
    """
    problem_name (str): name of problem for graph
    problem_df (DataFrame): should have err_scale, const_sample and nh_scale grouped by so one entry per combo. Regarding a single problem
    err_scale_list (list): list of all the floats used for error scalars in dataframe. Can include None.
    """
    best_rate_param_list = []

    # Add new column for best average hv and reciprocal gd to evals 
    problem_df['av_rec_gd+_to_evals'] = (1/problem_df["gd_plus"])/problem_df["evals"] 
    problem_df['av_hv_to_evals'] = (problem_df["hv"])/problem_df["evals"] 

    row_names = [row.name for _,row in problem_df.iterrows()]
    for i,err in enumerate(err_scale_list):
        # gets dataframe for each error as looping through each error in new colour/set
        row_names_filtered = list(filter(lambda x: x[0]==str(err), row_names))
        e_df = problem_df.loc[row_names_filtered]
        max_df = e_df["av_rec_gd+_to_evals"].idxmax()

        best_rate_param_list.append(list(max_df))
        # Because it is already grouped by, gives a tuple of the best values for max ratio!
        # print(f"{max_df[0]}:  const_sample={max_df[1]}  nbh_scale={max_df[2]}")
    
    return best_rate_param_list

def best_gd_and_hyper_overall(best_df_filename, problem_df, problem_name):
    """
    Get best average value over all parameter combos and append to file

    best_df_filename (str): Name of file to write/append findings to
    problem_df (DataFrame): should have err_scale, const_sample and nh_scale grouped by so one entry per combo. Regarding a single problem
    problem_name (str): name of problem for file
    """
    min_gd_plus_df = problem_df.loc[problem_df["gd_plus"].idxmin()]
    max_hv_df = problem_df.loc[problem_df["hv"].idxmax()]
    combined_df = pd.concat([min_gd_plus_df.to_frame().T, max_hv_df.to_frame().T])

    with open(best_df_filename, "a") as best_df_file:
        best_df_file.write(problem_name+"\n"+combined_df.to_string()+"\n\n")


if __name__=="__main__":

    # Scatter graphs
    problems = ["RE24", "RE31", "RE32", "RE33", "RE34", "RE37"]
    without_hv_problems = ["RE41", "RE42", "RE61", "RE91"]

    for prob in problems:
        RE_df = df_table_for_all_err_scalar(f"output/metric_data/{prob}_experiments.csv")
        scatter_graph_evals_per_gd_plus(prob, RE_df)
        scatter_graph_evals_per_hv(prob, RE_df)

    # More test examples
    # best_rate_param_list_per_prob = []
    # for prob in problems:
    #     RE_df = df_table_for_all_err_scalar(f"output/metric_data/{prob}_experiments.csv")
    #     best_rate_param_list = max_evals_per_metric_params(prob, RE_df)
    #     best_rate_param_list_per_prob.append(best_rate_param_list)

    # for prob in problems:
    #     RE_df = df_table_for_all_err_scalar(f"output/metric_data/{prob}_experiments.csv")
    #     best_gd_and_hyper_overall("best_df_per_prob.txt",RE_df,prob)

    # for prob in without_hv_problems:
    #     RE_df = df_table_for_all_err_scalar(f"output/metric_data/{prob}_experiments.csv")
    #     scatter_graph_evals_per_gd_plus(prob, RE_df)
    
    # for prob in without_hv_problems:
    #     RE_df = df_table_for_all_err_scalar(f"output/metric_data/{prob}_experiments.csv")
    #     best_gd_and_hyper_overall("best_df_per_prob.txt",RE_df,prob)

    # RE_df_31 = df_table_for_all_err_scalar(f"output/metric_data/RE31_experiments.csv")
    # best_gd_and_hyper_overall("RE31", RE_df_31)


