# gives statistics from a given dataset of numbers
import statistics

def root_mean_square(arr):
    sum = 0
    for n in arr:
        sum += n**2
    sum /= len(arr)
    return sum**0.5


def arithmetic_mean(arr):
    sum = 0
    for n in arr:
        sum += n
    sum /= len(arr)
    return sum

def geometric_mean(arr):
    sum = 1
    for n in arr:
        sum *= n
    return sum ** (1/len(arr))

def harmonic_mean(arr):
    sum = 0
    for n in arr:
        sum += n ** -1
    sum /= len(arr)
    return sum ** -1

def generalized_mean(arr, power):
    sum = 0
    for n in arr:
        sum += n**power
    sum /= len(arr)
    return sum**(1/n)

def median(arr):
    return statistics.median(arr)

def upper_quartile_median(arr):
    return statistics.median_high(arr)

def lower_quartile_median(arr):
    return statistics.median_low(arr)

def range(arr):
    return max(arr) - min(arr)

def mid_range(arr):
    return range(arr)/2

def standard_deviation(arr):
    average = arithmetic_mean(arr)
    sum = 0
    for n in arr:
        sum += (n - average) ** 2
    sum /= len(arr)
    return sum ** 0.5

def generalized_standard_deviation(arr, power):
    average = arithmetic_mean(arr)
    sum = 0
    for n in arr:
        sum += (n - average) ** power
    sum /= len(arr)
    return sum ** (1/power)
