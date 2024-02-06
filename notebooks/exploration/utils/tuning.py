from hyperopt import Trials, fmin


def perform_trials(objective_function, space, algo, max_evals):
    trials = Trials()
    best = fmin(fn=objective_function, space=space, algo=algo, max_evals=max_evals, trials=trials)

    return best, trials