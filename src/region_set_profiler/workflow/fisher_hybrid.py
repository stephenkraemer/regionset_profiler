from scipy.stats import fisher_exact
import numpy as np

%time fisher_exact(np.array([[10000, 200000], [5000, 8000]]))
