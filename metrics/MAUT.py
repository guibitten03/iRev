def calc(mse, rmse, mae, prec, rec, ndcg, div, nov):
    alg = ["convmf", "deepconn", "transnet", "dattn", "narre", "alfm", "tarmf", "a3ncf", "mpcn", "anr", "carl", "daml", "carp", "hrdr", "carm", "man"]
    values = sorted([i/16 for i in range(1,17)], reverse=True)

    mse = mse.split("&")
    mse = [x.strip() for x in mse]
    mse_rank = list(zip(alg, mse))
    mse_list = sorted(mse_rank, key=lambda x: x[1])
    mse_list = [x[0] for x in mse_list]
    mse_rank = {alg:score for alg, score in zip(mse_list, values)}

    rmse = rmse.split("&")
    rmse = [x.strip() for x in rmse]
    rmse_rank = list(zip(alg, rmse))
    rmse_list = sorted(rmse_rank, key=lambda x: x[1])
    rmse_list = [x[0] for x in rmse_list]
    rmse_rank = {alg:score for alg, score in zip(rmse_list, values)}

    mae = mae.split("&")
    mae = [x.strip() for x in mae]
    mae_rank = list(zip(alg, mae))
    mae_list = sorted(mae_rank, key=lambda x: x[1])
    mae_list = [x[0] for x in mae_list]
    mae_rank = {alg:score for alg, score in zip(mae_list, values)}

    prec = prec.split("&")
    prec = [x.strip() for x in prec]
    prec_rank = list(zip(alg, prec))
    prec_list = sorted(prec_rank, key=lambda x: x[1], reverse=True)
    prec_list = [x[0] for x in prec_list]
    prec_rank = {alg:score for alg, score in zip(prec_list, values)}

    rec = rec.split("&")
    rec = [x.strip() for x in rec]
    rec_rank = list(zip(alg, rec))
    rec_list = sorted(rec_rank, key=lambda x: x[1], reverse=True)
    rec_list = [x[0] for x in rec_list]
    rec_rank = {alg:score for alg, score in zip(rec_list, values)}

    ndcg = ndcg.split("&")
    ndcg = [x.strip() for x in ndcg]
    ndcg_rank = list(zip(alg, ndcg))
    ndcg_list = sorted(ndcg_rank, key=lambda x: x[1], reverse=True)
    ndcg_list = [x[0] for x in ndcg_list]
    ndcg_rank = {alg:score for alg, score in zip(ndcg_list, values)}

    div = div.split("&")
    div = [x.strip() for x in div]
    div_rank = list(zip(alg, div))
    div_list = sorted(div_rank, key=lambda x: x[1], reverse=True)
    div_list = [x[0] for x in div_list]
    div_rank = {alg:score for alg, score in zip(div_list, values)}

    nov = nov.split("&")
    nov = [x.strip() for x in nov]
    nov_rank = list(zip(alg, nov))
    nov_list = sorted(nov_rank, key=lambda x: x[1], reverse=True)
    nov_list = [x[0] for x in nov_list]
    nov_rank = {alg:score for alg, score in zip(nov_list, values)}

    error_maut = {}
    for alg, score in mse_rank.items():
        error_maut[alg] = score + rmse_rank[alg] + mae_rank[alg]

    rank_maut = {}
    for alg, score in prec_rank.items():
        rank_maut[alg] = score + rec_rank[alg] + ndcg_rank[alg]

    div_maut = {}
    for alg, score in div_rank.items():
        div_maut[alg] = score + nov_rank[alg]

    return error_maut, rank_maut, div_maut