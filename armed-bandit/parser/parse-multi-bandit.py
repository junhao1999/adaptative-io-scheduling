'''
----------------------------------------------------------------------------------------------------
MULTI ARMED-BANDIT SIMULATION PARSER
----------------------------------------------------------------------------------------------------

EUROPAR 2019 - Adaptive Request Scheduling for the I/O Forwarding Layer

----------------------------------------------------------------------------------------------------

AUTHORS
	Jean Luca Bez <jeanlucabez (at) gmail.com>
	Francieli Zanon Boito <francielizanon (at) gmail.com>

DESCRIPTION
	This file parses the simulation results of the armed bandit with a multiple access pattern.
	Notice that for now it only supports exactly three patterns at once.

CONTRIBUTORS
	Federal University of Rio Grande do Sul (UFRGS)
	INRIA France

NOTICE
	This program is distributed in the hope that it will be useful,
	but WITHOUT ANY WARRANTY; without even the implied warranty of
	MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.

CREATED
	December 13th, 2018

VERSION
	1.0

----------------------------------------------------------------------------------------------------
'''

import os
import os.path
import re
import sys
import csv
import subprocess
import collections
import logging
import pprint
import random
import time
import csv
import datetime
import json
import shlex
import errno
import argparse

import numpy as np

if sys.version_info[0] < 3:
	print('You must use Python 3')

	exit()

logging.basicConfig(
	level = logging.INFO,
	format = '%(asctime)s - %(levelname)s - %(message)s',
	handlers = [
		logging.FileHandler('parse.log'),
		logging.StreamHandler()
	]
)


def getBestSolution(metrics):
	# Get information about each window size performance and find out what is the best, return the best's performance and the best action
	best_choice = [-1, -1, -1]
	maximum_performance = [0, 0, 0]

	for pattern in metrics:
		p = int(pattern)

		for action, performance in metrics[pattern].items():
			if performance > maximum_performance[p]:
				maximum_performance[p] = performance
				best_choice[p] = action

	for i in range(len(best_choice)):
		if best_choice[i] == -1 :
			logging.error('Unable to find the best solution')

			exit()

	ret = []

	for pattern in range(len(best_choice)):
		ret.append(metrics[str(pattern)][best_choice[pattern]])
	return (ret, best_choice)


def getMaximumIterations(filename):
	# Get the number of iterations, total reward from the JSON
	with open('{}'.format(filename)) as f:
		data = json.load(f)

	return data['bandit']['iterations']


def getIterations(filename, maximum_iterations, BIN_SIZE, best_choice):
	# Read maximum_iterations lines, separate measurements in bins of BIN_SIZE
	# Return two lists, the first has the maximum_iterations/BIN_SIZE average rewards, the second has the precision of each bin
	iteration = 0

	series = []
	precision = []
	current_bin = [[], [], []]

	took_best = [0, 0, 0]

	f = open(filename.replace('.json', '-log.csv'), 'r')
		
	# Skip the header
	f.readline()

	while iteration < maximum_iterations:
		line = f.readline()
		parsed = line.split('\n')[0].split(';')
		pattern = int(parsed[1])

		# Time to close a bin
		if (iteration > 0) and ((iteration % BIN_SIZE) == 0):
			this_series = []
			this_precision = []

			for p in range(len(current_bin)):
				if len(current_bin[p]) == 0:
					this_series.append(-1)
					this_precision.append(-1)
				else:
					this_series.append(np.mean(current_bin[p]))
					this_precision.append(float(took_best[p]) / len(current_bin[p]))
					
			series.append(this_series)
			precision.append(this_precision)

			# Reset the counter and the current bin
			took_best = [0, 0, 0]
			current_bin = [[], [], []]

		current_bin[pattern].append(float(parsed[3]))

		# Count the number of times we took the best choice
		if int(parsed[2]) == int(best_choice[pattern]):
			took_best[pattern] += 1

		iteration += 1

	f.close()

	return (series, precision)
		

def getResults(filename, maximum_iterations, BIN_SIZE):
	# Get the number of iterations, total reward from the JSON
	with open('{}'.format(filename)) as f:
		data = json.load(f)

	iterations = data['bandit']['iterations']

	if iterations < maximum_iterations:
		logging.error('{} - Not enough iterations!'.format(filename))

		exit()

	# From the part with the performance of different windows, find the best and see how many times it took it
	best_performance, best_choice = getBestSolution(data['bandit']['reward']['summary'])

	# From the log of the iterations, fill the bins until the last iteration we will consider
	series, precision = getIterations(filename, maximum_iterations, BIN_SIZE, best_choice)
	
	# Normalize bandwidth by the best possible
	normalized = []

	for o in range(len(series)):
		normalized.append([])
		
		if len(series[o]) != 3:
			logging.error('Something weird with read bins')
			logging.error(series)
	
			exit(1)

		for p in range(len(series[o])):
			normalized[o].append(series[o][p] / best_performance[p])

	# Return this repetition's series of normalized average rewards and precision
	return (normalized, precision)

	
def getFiles(io_nodes, processes, test_type, strided, operation, request_size, epsilon, alpha):
	if alpha is None:
		ret = subprocess.getoutput('ls results/multiple/epsilon-{}*.json | grep -v alpha'.format(epsilon)).split('\n')
	else:
		ret = subprocess.getoutput('ls results/multiple/epsilon-{}-alpha-{}*.json'.format(epsilon, alpha)).split('\n')

	if 'found' in ret[0]:
		logging.error('Unable to find the access pattern')
		logging.error(ret)

		exit()

	return ret

#########################################################################################################################


# Configure the parser
parser = argparse.ArgumentParser()

parser.add_argument('--ions',
	action = 'store',
	dest = 'io_nodes',
	required = True,
	help = 'Number of I/O nodes')

parser.add_argument('--processes',
	action = 'store',
	dest = 'processes',
	required = True,
	help = 'Number of processes')

parser.add_argument('--type',
	action = 'store',
	dest = 'test_type',
	required = True,
	help = 'Type: (1) file-per-process or (2) shared-file')

parser.add_argument('--strided',
	action = 'store',
	dest = 'strided',
	required = True,
	help = 'Spatiality: (0) contiguous or (1) 1D-strided')

parser.add_argument('--size',
	action = 'store',
	dest = 'request_size',
	required = True,
	help = 'Request size')

parser.add_argument('--operation',
	action = 'store',
	dest = 'operation',
	required = True,
	help = 'Operation: "read" or "write"')

parser.add_argument('--directory',
	action = 'store',
	dest = 'directory',
	required = True,
	help = 'Directory to store the CSVs')


args = parser.parse_args()

BIN_SIZE = 10
EPSILA = [1, 3, 5, 7, 10, 15, 20]
ALPHA = [None]

files = []

for epsilon in EPSILA:
	for alpha in ALPHA:
		experiments = getFiles(args.io_nodes, args.processes, args.test_type, args.strided, args.operation, args.request_size, epsilon, alpha)
		files.append([epsilon, alpha, experiments])

output_file = open('{}/{}-{}-{}-{}-{}-{}.csv'.format(args.directory, args.io_nodes, args.processes, args.test_type, args.strided, args.request_size, args.operation), 'w')
output_file.write("epsilon;alpha;pattern;bin;observation;performance;precision\n")

# Iterate over all the files
for f in range(len(files)):
	epsilon = files[f][0]
	alpha = files[f][1]

	performance = [[], [], []]
	correct_answer = [[], [], []]

	maximum_iterations = sys.maxsize

	# Get the maximum number of iterations based on the test files
	for filename in files[f][2]:
		maximum_iterations = min(maximum_iterations, getMaximumIterations(filename))
	
	logging.info('Maximum iterations: {}'.format(maximum_iterations))
	
	# Create the slots for the metrics
	for i in range(int(maximum_iterations / BIN_SIZE + 1)):
		for p in range(len(performance)):
			performance[p].append([])
			correct_answer[p].append([])

	for filename in files[f][2]:
		print(filename)
		normalized, precision = getResults(filename, maximum_iterations, BIN_SIZE)

		for i in range(len(normalized)):
			for p in range(len(normalized[i])):
				performance[p][i].append(normalized[i][p])
				correct_answer[p][i].append(precision[i][p])

	# Write output to the file
	for p in range(len(performance)):
		for i in range(len(performance[p])):
			for j in range(len(performance[p][i])):
				output_file.write('{};{};{};{};{};{};{}\n'.format(epsilon, alpha, p, i + 1, j + 1, performance[p][i][j], correct_answer[p][i][j]))

output_file.close()
