import pandas as pd
import numpy as np

def load_error_and_rank_metrics(algorithms, dataset, emb, lenght, path="results/", bert="default"):

    labels = ['MSE', 'MAE', 'RMSE', 'PRECISION', 
              'RECALL', 'DIVERSITY', 'NOVELTY', 'NDCG']
    
    alg_name_1 = [""]*len(labels) 
    alg_value_1 = [1.0]*len(labels[:3]) + [0.0]*len(labels[3:])

    alg_name_2 = [""]*len(labels)
    alg_value_2 = [1.0]*len(labels[:3]) + [0.0]*len(labels[3:])

    draw = []
    metrics_matrix = []
    
    for algorithm in algorithms:
            if bert != "default":
                algorithm = algorithm + "_" + bert
        
            error_metrics = pd.read_csv(
                path + algorithm + "_" + dataset + "_" + emb + "_results_error.csv")
            
            rank_metrics = pd.read_csv(
                path + algorithm + "_" + dataset + "_" + emb + "_results_rank.csv")
            

            values = np.concatenate([error_metrics.mean().values, 
                                    rank_metrics.mean().values])
            
            for i, m in enumerate(values[:3]):
                if m < alg_value_1[i]:
                    alg_name_1[i] = algorithm
                    alg_value_1[i] = m

                elif m < alg_value_2[i]:
                    alg_name_2[i] = algorithm
                    alg_value_2[i] = m

            for i, m in enumerate(values[3:]):
                idx = i + 3
                if m > alg_value_1[idx]:
                    alg_name_1[idx] = algorithm
                    alg_value_1[idx] = m

                elif m > alg_value_2[idx]:
                    alg_name_2[idx] = algorithm
                    alg_value_2[idx] = m


            metrics_matrix.append(values)

    # for i, (alg1, alg2) in enumerate(zip(alg_name_1, alg_name_2)):
        
    #     alg1_metrics_error = pd.read_csv(
    #             path + alg1 + "_" + dataset + "_" + emb + "_results_error.csv").values.transpose()
            
    #     alg1_metrics_rank = pd.read_csv(
    #             path + alg1 + "_" + dataset + "_" + emb + "_results_rank.csv").values.transpose()
        
    #     alg2_metrics_error = pd.read_csv(
    #             path + alg2 + "_" + dataset + "_" + emb + "_results_error.csv").values.transpose()
            
    #     alg2_metrics_rank = pd.read_csv(
    #             path + alg2 + "_" + dataset + "_" + emb + "_results_rank.csv").values.transpose()
        
    #     if i < 7:
    #         alg1_metric = alg1_metrics_error[i]
    #         alg2_metric = alg2_metrics_error[i]
    #     else:
    #         alg1_metric = alg1_metrics_rank[0]
    #         alg2_metric = alg2_metrics_rank[0]

    #     if len(alg1_metric) > lenght:
    #         alg1_metric = regulate_metrics(alg1_metric, lenght)
    #     if len(alg2_metric) > lenght:
    #         alg2_metric = regulate_metrics(alg2_metric, lenght)

    #     if not (np.array(alg1_metric) - np.array(alg2_metric)).any():
    #         test = 0.0
    #     else:
    #         test = wilcoxon(alg1_metric, alg2_metric).pvalue
            
    #     if test < 0.05:
    #         draw.append('Draw')
    #     else:
    #         draw.append('Winner')


    df = pd.DataFrame(np.array(metrics_matrix),
                      columns=labels, index=algorithms)
    
    # rank = pd.DataFrame({"Metric": labels,
    #                     "Winner":alg_name_1,
    #                      "2 Place": alg_name_2,
    #                      "Draw": draw})
    rank = []

    return df.round(3), rank
    
    # return  error_metrics.mean(), rank_metrics.mean()