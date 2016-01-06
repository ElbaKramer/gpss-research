Experiment(description='Trying latest code on extrapolation task',
           data_dir='../examples/data',
           max_depth=5, 
           random_order=False,
           k=1,
           debug=False, 
           local_computation=False, 
           n_rand=9,
           sd=2, 
           jitter_sd=0.1,
           max_jobs=600, 
           verbose=False,
           make_predictions=False,
           skip_complete=True,
           results_dir='../examples/results',
           iters=250,
           base_kernels='SE,Per,Lin,Const,Noise',
           random_seed=1,
           period_heuristic=3,
           period_heuristic_type='min',
           subset=True,
           subset_size=250,
           full_iters=10,
           bundle_size=5,
           additive_form=True,
           mean='ff.MeanZero()',      # Starting mean
           kernel='ff.NoiseKernel()', # Starting kernel
           lik='ff.LikGauss(sf=-np.Inf)', # Starting likelihood 
           score='bic',
           relational=True,
           search_operators=[('A', ('+', 'A', 'B'), {'A': 'kernel', 'B': 'base'}),
                             ('A', ('*', 'A', 'B'), {'A': 'kernel', 'B': 'base-not-const'}),
                             ('A', ('*-const', 'A', 'B'), {'A': 'kernel', 'B': 'base-not-const'}),
                             ('A', 'B', {'A': 'kernel', 'B': 'base'}),
                             ('A', ('CP', 'd', 'A'), {'A': 'kernel', 'd' : 'dimension'}),
                             ('A', ('CW', 'd', 'A'), {'A': 'kernel', 'd' : 'dimension'}),
                             ('A', ('B', 'd', 'A'), {'A': 'kernel', 'd' : 'dimension'}),
                             ('A', ('BL', 'd', 'A'), {'A': 'kernel', 'd' : 'dimension'}),
                             ('A', ('None',), {'A': 'kernel'})])

