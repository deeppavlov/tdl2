import math


def visualize_margin_distributions(trainers:List[Trainer], max_n_cols:int=3, mode:str='train'):
    # TODO: is it possible to plot several hists in one subplot?
    n_plots = len(trainers)
    n_cols = min(max_n_cols, len(trainers))
    n_rows = math.ceil(n_plots / n_cols)
    _, subplots = plt.subplots(n_rows, n_cols, figsize=(n_cols * 6, n_rows * 4))
    if n_rows > 1: subplots = [p for row in subplots for p in row]
    subplots = subplots[:n_plots]
    
    for t, subplot in zip(trainers, subplots):
        sc = compute_spectral_complexity(t)[0].value
        dataloader = t.train_dataloader if mode == 'train' else t.test_dataloader
        dist = compute_margin_dist(trainer.model, dataloader, sc)
        
        subplot.set_title(f'Proportion of bad points ({mode}): {t.bad_points_proportion}')
        subplot.hist(dist, bins=100)
        subplot.grid()
        