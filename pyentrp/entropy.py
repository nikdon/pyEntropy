# -*- coding: utf-8 -*-

from __future__ import unicode_literals

import itertools
import numpy as np

def util_hash_term(perm):
    """Associate unique integer to a permutation.
    
    Args:
        perm: list of permutation elements. Each element in perm must be integer and <= N, where N = len(perm). For example [2,0,1] or [2,3,0,1].
        
    Returns:  
        int
    
    Added by Jakob Dreyer, 2018, Dept Bioinformatics, H Lundbeck A/S, Denmark"""
    
    deg = len(perm)
    return sum([perm[k]*deg**k for k in range(deg)])


def util_pattern_space(time_series, lag, dim):
    """Create a set of sequences with given lag and dimension

    Args:
       time_series: Vector or string of the sample data
       lag: Lag between beginning of sequences
       dim: Dimension (number of patterns)

    Returns:
        2D array of vectors

    """
    n = len(time_series)

    if lag * dim > n:
        raise Exception('Result matrix exceeded size limit, try to change lag or dim.')
    elif lag < 1:
        raise Exception('Lag should be greater or equal to 1.')

    pattern_space = np.empty((n - lag * (dim - 1), dim))
    for i in range(n - lag * (dim - 1)):
        for j in range(dim):
            pattern_space[i][j] = time_series[i + j * lag]

    return pattern_space


def util_standardize_signal(time_series):
    return (time_series - np.mean(time_series)) / np.std(time_series)


def util_granulate_time_series(time_series, scale):
    """Extract coarse-grained time series

    Args:
        time_series: Time series
        scale: Scale factor

    Returns:
        Vector of coarse-grained time series with given scale factor
    """
    n = len(time_series)
    b = int(np.fix(n / scale))
    temp = np.reshape(time_series[0:b*scale], (b, scale))
    cts = np.mean(temp, axis = 1)
    return cts


def shannon_entropy(time_series):
    """Return the Shannon Entropy of the sample data.

    Args:
        time_series: Vector or string of the sample data

    Returns:
        The Shannon Entropy as float value
    """

    # Check if string
    if not isinstance(time_series, str):
        time_series = list(time_series)

    # Create a frequency data
    data_set = list(set(time_series))
    freq_list = []
    for entry in data_set:
        counter = 0.
        for i in time_series:
            if i == entry:
                counter += 1
        freq_list.append(float(counter) / len(time_series))

    # Shannon entropy
    ent = 0.0
    for freq in freq_list:
        ent += freq * np.log2(freq)
    ent = -ent

    return ent


def sample_entropy(time_series, m, tol = None):
    """Calculates the sample entropy of degree m of a time_series.
    
    This method uses chebychev norm and searches iteratively. It is quite fast for random data, but can be slower is there is structure in the time series. 
    
    Args:
        time_series: numpy array of time series
        m: length of template vector
        r: tolerance (defaults to 0.1 * std(time_series)))
    Returns: 
        Sample entropy (float)
        
    References:
        [1] http://en.wikipedia.org/wiki/Sample_Entropy
        [2] http://physionet.incor.usp.br/physiotools/sampen/
        [3] Madalena Costa, Ary Goldberger, CK Peng. Multiscale entropy analysis
            of biological signals
            """
        
    if tol is None:
        tol = 0.1*np.std(time_series)

    n = len(time_series)
    Ntemp = 0;
    Ntemp_plus = 0;
    for i in range(n-m-1):
        template = time_series[i:(i+m+1)];#We have 'm+1' elements in the template
        rem_time_series = time_series[i+1:]
        
        """Search for matches in remtime_series. First we look for match with template[0]. 
        We then select the neighbors of these that also match next elements """
        
        "Create a list with those indices in the time series that match the first element in template"
        searchlist = np.nonzero(np.abs(rem_time_series - template[0]) < tol)[0]

        go = len(searchlist) > 0;
        
        length = 1;
       
        if length == m:
            Ntemp += len(searchlist)
        
        """This while loop keeps reducing the searchlist by also comparing next elements making the templtates longer. 
        It will stop if there are no elements that match or if we reach m+1-length-templates. """
        while go:
            
            "Shift the index 1 step to the right in order to find elements in the time series next to the ones we found:"
            nextindxlist = searchlist + 1;
            "remove elements to close to the end:"
            nextindxlist = nextindxlist[nextindxlist < n - 1 - i]
           
            "This is the list of elements that we shopuld compare with the next entry in the template vector:"
            nextcandidates = rem_time_series[nextindxlist]
            
            "Hitlist is bool list and true where next time_series elements match next template elements"
            hitlist = np.abs(nextcandidates - template[length]) < tol
            "reduce the search list to those elements that surviced this round"
            searchlist = nextindxlist[hitlist]
           
            if length == m-1:
                Ntemp += sum(hitlist)
            elif length == m :
                Ntemp_plus += sum(hitlist)
            
           
            length += 1;
            go = any(hitlist) and length < m + 1    
            
            
    #print('Ntemp m', Ntemp)
    #print('Ntemp m+1', Ntemp_plus)
    if Ntemp > 0:
        sampen =  - np.log(Ntemp_plus/Ntemp)       
    else:
        sampen = np.inf
        
    return sampen



def multiscale_entropy(time_series, sample_length, tolerance):
    """Calculate the Multiscale Entropy of the given time series considering
    different time-scales of the time series.

    Args:
        time_series: Time series for analysis
        sample_length: Bandwidth or group of points
        tolerance: Tolerance (default = 0.1...0.2 * std(time_series))

    Returns:
        Vector containing Multiscale Entropy

    Reference:
        [1] http://en.pudn.com/downloads149/sourcecode/math/detail646216_en.html
    """
    n = len(time_series)
    mse = np.zeros((1, sample_length))

    for i in range(sample_length):
        b = int(np.fix(n / (i + 1)))
        temp_ts = [0] * int(b)
        for j in range(b):
            num = sum(time_series[j * (i + 1): (j + 1) * (i + 1)])
            den = i + 1
            temp_ts[j] = float(num) / float(den)
        se = sample_entropy(temp_ts, 1, tolerance)
        mse[0, i] = se

    return mse[0]


def permutation_entropy(time_series, m, delay):
    """Calculate the Permutation Entropy

    Args:
        time_series: Time series for analysis
        m: Order of permutation entropy
        delay: Time delay

    Returns:
        Vector containing Permutation Entropy

    Reference:
        [1] Massimiliano Zanin et al. Permutation Entropy and Its Main Biomedical and Econophysics Applications:
            A Review. http://www.mdpi.com/1099-4300/14/8/1553/pdf
        [2] Christoph Bandt and Bernd Pompe. Permutation entropy — a natural complexity
            measure for time series. http://stubber.math-inf.uni-greifswald.de/pub/full/prep/2001/11.pdf
        [3] http://www.mathworks.com/matlabcentral/fileexchange/37289-permutation-entropy/content/pec.m
    """
    n = len(time_series)
    permutations = np.array(list(itertools.permutations(range(m))))
    hashlist = [util_hash_term(perm) for perm in permutations]
    c = [0] * len(permutations)

    for i in range(n - delay * (m - 1)):
        # sorted_time_series =    np.sort(time_series[i:i+delay*m:delay], kind='quicksort')
        sorted_index_array = np.array(np.argsort(time_series[i:i + delay * m:delay], kind='quicksort'))
        hashvalue = util_hash_term(sorted_index_array);
        c[np.argwhere(hashlist == hashvalue)[0][0]] += 1

    c = [element for element in c if element != 0]
    p = np.divide(np.array(c), float(sum(c)))
    pe = -sum(p * np.log(p))
    return pe


def multiscale_permutation_entropy(time_series, m, delay, scale):
    """Calculate the Multiscale Permutation Entropy

    Args:
        time_series: Time series for analysis
        m: Order of permutation entropy
        delay: Time delay
        scale: Scale factor

    Returns:
        Vector containing Multiscale Permutation Entropy

    Reference:
        [1] Francesco Carlo Morabito et al. Multivariate Multi-Scale Permutation Entropy for
            Complexity Analysis of Alzheimer’s Disease EEG. www.mdpi.com/1099-4300/14/7/1186
        [2] http://www.mathworks.com/matlabcentral/fileexchange/37288-multiscale-permutation-entropy-mpe/content/MPerm.m
    """
    mspe = []
    for i in range(scale):
        coarse_time_series = util_granulate_time_series(time_series, i + 1)
        pe = permutation_entropy(coarse_time_series, m, delay)
        mspe.append(pe)
    return mspe


# TODO add tests
def composite_multiscale_entropy(time_series, sample_length, scale, tolerance=None):
    """Calculate the Composite Multiscale Entropy of the given time series.

    Args:
        time_series: Time series for analysis
        sample_length: Number of sequential points of the time series
        scale: Scale factor
        tolerance: Tolerance (default = 0.1...0.2 * std(time_series))

    Returns:
        Vector containing Composite Multiscale Entropy

    Reference:
        [1] Wu, Shuen-De, et al. "Time series analysis using
            composite multiscale entropy." Entropy 15.3 (2013): 1069-1084.
    """
    cmse = np.zeros((1, scale))

    for i in range(scale):
        for j in range(i):
            tmp = util_granulate_time_series(time_series[j:], i + 1)
            cmse[i] += sample_entropy(tmp, sample_length, tolerance) / (i + 1)

    return cmse
