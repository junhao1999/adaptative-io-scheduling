'''
----------------------------------------------------------------------------------------------------
MULTI ARMED-BANDIT SIMULATION
----------------------------------------------------------------------------------------------------

EUROPAR 2019 - Adaptive Request Scheduling for the I/O Forwarding Layer

----------------------------------------------------------------------------------------------------

AUTHORS
	Jean Luca Bez <jeanlucabez (at) gmail.com>
	Francieli Zanon Boito <francielizanon (at) gmail.com>

DESCRIPTION
	This file simulates the armed bandit with a multiple access pattern.

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


def parseObservation(line):
	# 0  ;1    ;2          ;3           ;4             ;5             ;6      ;7    ;8        ;
	# row;label;observation;pvfs_servers;iofsl_servers;iofsl_server_id;clients;twins;processes;global_total_request_number;global_total_read_number;global_total_write_number;global_avg_request_time;global_min_request_time;global_max_request_time;global_avg_read_request_time;global_avg_write_request_time;global_avg_request_size;global_min_request_size;global_max_request_size;global_avg_read_request_size;global_avg_write_request_size;file_handles;queue_server_1;queue_server_2;queue_server_3;queue_server_4;operation;received_request_number;processed_request_number;processed_request_size;processed_sum_request_time;avg_received_size;sum_request_size;min_request_size;max_request_size;avg_request_time;min_request_time;max_request_time;avg_distance;benchmark_bandwidth;median_processed_bandwidth;processed_sum_request_bandwidth;processed_avg_request_bandwidth;observation_window_bandwidth
	parsed = line.split('\n')[0].split(';')

	observation = {}
	observation['io_nodes'] = int(parsed[4])
	observation['processes'] = int(parsed[8])
	observation['operation'] = parsed[27].upper()
	observation['bandwidth'] = float(parsed[len(parsed)-1])
	observation['window'] = int(parsed[7])
	observation['observation'] = int(parsed[2])
	observation['label'] = parsed[1]

	return observation


def getExperimentLabelsPattern(io_nodes, processes, test_type, strided, operation, request_size):
	labels = {}

	# Open the file with the configuration and results of all the experiments
	f = open('../experiments/experiment-025-iofsl-4gb-4pvfs-twins-metrics-agios/model/experiments-results.csv', 'r')

	for line in f:
		if 'repetition' in line:
			continue

		experiment = line.split('\n')[0].split(';')

		# 0  ;1         ;2      ;3            ;4           ;5    ;6           ;7           ;8        ;9        ;10       ;11       ;12     ;13  ;14  ;	
		# row;repetition;clients;iofsl_servers;pvfs_servers;label;iofsl_config;agios_config;scheduler;processes;test_type;operation;strided;nobj;size;total_bandwidth;total_time;bandwidth;time;MPI_File_Open_Time;MPI_File_Wait_Time;MPI_File_Preallocate_Time;MPI_File_Close_Wait_Time;computed_bandwidth

		if (int(experiment[3]) == int(io_nodes)) and (int(experiment[9]) == int(processes)) and (int(experiment[10]) == int(test_type)) and (experiment[11].upper() == operation) and (int(experiment[12]) == int(strided)) and (int(experiment[14]) == int(request_size)):
			window_size = int(experiment[8].split('-')[1].split('U')[0])

			if not window_size in labels:
				labels[window_size] = []

			labels[window_size].append(experiment[5])

	f.close()

	logging.info('Selected experiments:')

	flat = []

	for window_size in labels:
		logging.info('TWINS {:4d}(ms) - {}'.format(window_size, labels[window_size]))

		for id in labels[window_size]:
			flat.append(id)

	return flat

def getExperimentLabels(io_nodes, processes, test_type, strided, operation, request_size):
	ret = []

	for i in range(len(io_nodes)):
		ret.append(getExperimentLabelsPattern(io_nodes[i], processes[i], test_type[i], strided[i], operation[i], request_size[i]))

	return ret


def getID(label):
	return label.split('-')[1]


def getObservationsPattern(io_nodes, processes, operation, labels):
	# See how many lines we have to skip
	mark = {}

	f = open('../experiments/experiment-025-iofsl-4gb-4pvfs-twins-metrics-agios/model/experiments-marks.csv', 'r')

	for line in f:
		if 'operation' in line:
			continue

		test = line.split('\n')[0].split(';')

		if (test[1] in labels) and (operation == test[2].upper()):
			key = '{}-{}'.format(test[1], test[2].upper())

			mark[key] = {}
			mark[key]['start'] = int(test[3])
			mark[key]['end'] = int(test[4])

	f.close()

	logging.info('Skip lines: {}'.format(mark))

	# Read the actual dataset
	entries = {}

	f = open('../experiments/experiment-025-iofsl-4gb-4pvfs-twins-metrics-agios/model/experiments-bandwidth-full.csv', 'r')

	selected = 0
	wrong_label = 0
	wrong_ap = 0
	skipped = 0

	for line in f:
		if 'label' in line:
			continue

		if line.split(';')[1] in labels:
			parsed = parseObservation(line)

			if (int(parsed['io_nodes']) == int(io_nodes)) and (int(parsed['processes']) == int(processes)) and (parsed['operation'].upper() == operation.upper()):
				# We need to ignore the metrics with zero reported bandwith, as that would not be passed on to the council
				if parsed['bandwidth'] > 0.0 and parsed['observation'] >= mark['{}-{}'.format(parsed['label'], parsed['operation'])]['start'] and parsed['observation'] <= mark['{}-{}'.format(parsed['label'], parsed['operation'])]['end']:
					if not (parsed['window'] in entries):
						entries[parsed['window']] = []

					selected += 1
					entries[parsed['window']].append(parsed['bandwidth'])
				else:
					skipped += 1
			else:
				wrong_ap +=1
		else:
			wrong_label += 1

	logging.info('Skip wrong labels: {} observations'.format(wrong_label))
	logging.info('Skip wrong pattern: {} observations'.format(wrong_ap))
	logging.info('Skip interval: {} observations'.format(skipped))
	logging.info('Selected: {} observations'.format(wrong_label))

	logging.info('Observations:')

	for window in entries:
		logging.info('TWINS {:4d}(ms) - {} observations'.format(window, len(entries[window])))

	return entries


def getObservations(io_nodes, processes, operation, labels):
	ret = []

	for i in range(len(operation)):
		ret.append(getObservationsPattern(io_nodes[i], processes[i], operation[i], labels[i]))
	return ret


def getBestQ(Q):
	best = 0

	# Find the best, i.e., maximum value
	for action in Q:
		best = max(best, Q[action])

	choices = []

	# Get the choices with that value
	for action in Q:
		if Q[action] == best:
			choices.append(action)

	# Randomly return one of the good choices
	return choices[random.randint(0, len(choices) - 1)]


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

parser.add_argument('--epsilon',
	action = 'store',
	dest = 'epsilon',
	required = True,
	help = 'Armed Bandit: epsilon')

parser.add_argument('--alpha',
	action = 'store',
	dest = 'alpha',
	required = False,
	help = 'Armed Bandit: alpha')

parser.add_argument('--output',
	action = 'store',
	dest = 'output',
	required = True,
	help = 'Output file')

args = parser.parse_args()

# Make sure the operation is in uppercase
args.operation = args.operation.upper()

#random.seed(123456789)
#np.random.seed(123456789)

logging.basicConfig(
	level = logging.INFO,
	format = '%(asctime)s - %(levelname)s - %(message)s',
	handlers = [
		logging.FileHandler('logs/simulate-{}-{}-{}-{}-{}-{}.log'.format(args.io_nodes, args.processes, args.test_type, args.strided, args.operation, args.request_size)),
		logging.StreamHandler()
	]
)

# Get the label of the experiments that match the given configuration
labels = getExperimentLabels(args.io_nodes.split(','), args.processes.split(','), args.test_type.split(','), args.strided.split(','), args.operation.split(','), args.request_size.split(','))

# Get the metrics from those labels
observations = getObservations(args.io_nodes.split(','), args.processes.split(','), args.operation.split(','), labels)

if not observations:
	logging.error('No observations found!')

	exit()

# Actions are the window sizes (in miliseconds) that we allow TWINS to use
actions = [
	125,
	250,
	500,
	1000,
	2000,
	4000,
	8000
]

# Initialization of the Armed Bandit
Q = []
N = []

for o in range(len(args.operation.split(','))):
	Q.append({})
	N.append({})

	for a in actions:
		Q[o][a] = 0.0
		N[o][a] = 0

logging.info('Bandwidth:')

# Keep track of the bandwidth provided by the different actions
observed_bandwidth = []

for o in range(len(args.operation.split(','))):
	observed_bandwidth.append({})

	for action in actions:
		observed_bandwidth[o][action] = np.mean(observations[o][action])

		logging.info('PATTERN #{} TWINS {:4d}(ms) - {}MB/s'.format(o, action, observed_bandwidth[o][action]))

log = []

total = 0
iteration = 0

iteration_per_application = []

for ap in range(len(args.operation.split(','))):
	iteration_per_application.append(0)

# We keep testing until we have observations that were not already used
while True:
	# Randomly decide wthat is the access pattern coming now
	access_pattern = random.randint(0, len(args.operation.split(',')) - 1)

	# Determine if we should EXPLORE or EXPLOIT
	if random.randint(1, 100) <= int(args.epsilon):
		# EXPLORATION: take a random action
		action = actions[random.randint(0, len(actions) - 1)]
	else:
		# EXPLOITATION: take the best action
		action = getBestQ(Q[access_pattern])

	# Check if we have enough data to continue the simulation
	if len(observations[access_pattern][action]) == 0:
		logging.info('FINISHED - {} iterations'.format(iteration))

		break
	else:
		iteration += 1

		iteration_per_application[access_pattern] += 1

	# Take the action and get the reward, i.e. get a random observation for this action
	reward = observations[access_pattern][action][random.randint(0, len(observations[access_pattern][action]) - 1)]

	# Remove the observation to avoid repetition
	observations[access_pattern][action].remove(reward)

	total += reward

	log.append([access_pattern, action, reward, total])

	# Update our value estimation for this action
	N[access_pattern][action] += 1

	if args.alpha is None:
		# If we do not have an alpha defined we should the Incremental implementation
		Q[access_pattern][action] = Q[access_pattern][action] + (1.0 / float(N[access_pattern][action])) * (reward - Q[access_pattern][action])
	else:
		# If we have an alpha we should use the Nonstationary formula
		Q[access_pattern][action] = Q[access_pattern][action] + float(args.alpha) * (reward - Q[access_pattern][action])


# Generates the summary
summary = collections.OrderedDict()

summary['experiment'] = collections.OrderedDict()
summary['experiment']['io_nodes'] = (args.io_nodes.split(','))
summary['experiment']['processes'] = (args.processes.split(','))
summary['experiment']['test_type'] = (args.test_type.split(','))
summary['experiment']['strided'] = (args.processes.split(','))
summary['experiment']['operation'] = args.operation.split(',')
summary['experiment']['request_size'] = (args.request_size.split(','))

summary['bandit'] = collections.OrderedDict()
summary['bandit']['algorithm'] = 'sample-average' if args.alpha is None else 'exponential recency-weighted average'
summary['bandit']['epsilon'] = int(args.epsilon)
if args.alpha is not None:
	summary['bandit']['alpha'] = float(args.alpha)

summary['bandit']['iterations'] = iteration
summary['bandit']['pattern_iterations'] = iteration_per_application

summary['bandit']['reward'] = collections.OrderedDict()
summary['bandit']['reward']['total'] = total
summary['bandit']['reward']['average'] = total / iteration

summary['bandit']['reward']['summary'] = {}

for o in range(len(args.operation.split(','))):
	summary['bandit']['reward']['summary'][o] = {}

	for action in actions:
		summary['bandit']['reward']['summary'][o][action] = observed_bandwidth[o][action]

with open('{}.json'.format(args.output), 'w') as json_file:
	try:
		json.dump(summary, json_file, sort_keys = False, indent = 4)
		logging.info('summary file: {}.json'.format(args.output))
	except:
		logging.error('summary file: {}.json'.format(args.output))

# Generates the CSV
output = csv.writer(open('{}.csv'.format(args.output), 'w'), delimiter = ';')

header = [
    'pattern',
	'window',
	'times',
	'proportion',
	'Q'
]

# Write CSV header
output.writerow(header)

for o in range(len(args.operation.split(','))):
	for action in actions:
		output.writerow([
            o,
			action,
			N[o][action],
			float(N[o][action]) / iteration_per_application[o],
			Q[o][action]
		])

logging.info('CSV file: {}.csv'.format(args.output))

# Generates the LOG CSV
output = csv.writer(open('{}-log.csv'.format(args.output), 'w'), delimiter = ';')

header = [
	'iteration',
    'pattern',
	'action',
	'reward',
	'accumulated'
]

# Write CSV header
output.writerow(header)

for iteration, entry in enumerate(log):
	output.writerow([
		iteration,
		entry[0],
		entry[1],
		entry[2],
		entry[3]
	])

logging.info('CSV file: {}-log.csv'.format(args.output))