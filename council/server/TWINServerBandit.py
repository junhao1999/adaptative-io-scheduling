"""Defined the behavior of the TWINS Server Council."""

# -*- coding: utf-8 -*-

import os
import os.path
import datetime
import random
import sys
import socket
import select
import signal
import logging
import logging.handlers

import numpy as np
import pandas as pd

from sklearn.preprocessing import PowerTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

from keras.models import load_model
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import RMSprop


class TWINServerBandit:
    """TWINS Server Council."""

    MODEL_TRAINING_EPOCHS = 50
    MODEL_DATA_PATH = 'access-pattern-metrics-train.csv'
    MODEL_FILE_PATH = 'tf-model-keras-trained-2018-10-10.h5'

    LOG_FILENAME = 'twins-server.log'

    USE_NEURAL_NETWORK = True

    def __init__(self, debug, host, port, epsilon, alpha):
        """Configure the  logs, shutdown, server, and intialize the Council."""
        self.configure_log(debug)
        self.configure_signal()
        self.configure_server(host, port)
        self.train_model()
        self.initialize_bandits(epsilon, alpha)

    def configure_signal(self):
        """Signal the script to gracefully exit."""
        signal.signal(signal.SIGINT, self.signal_handler)

    def signal_handler(self, sig, frame):
        """Handle the signals we capture."""
        self.logger.info('Shutdown requested (SIGINT)')

        # Display the statistics of the server
        self.logger.info('N = {}'.format(self.N))
        self.logger.info('Q = {}'.format(self.Q))

        sys.exit(0)

    def configure_log(self, debug):
        """Configure the log system."""
        # Creates and configure the log file
        self.logger = logging.getLogger('TWINServerBandit')

        if debug:
            self.logger.setLevel(logging.DEBUG)
        else:
            self.logger.setLevel(logging.INFO)

        # Defines the format of the logger
        formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s'
        )

        # Configure the log rotation
        handler = logging.handlers.RotatingFileHandler(
            self.LOG_FILENAME,
            maxBytes=268435456,
            backupCount=50,
            encoding='utf8'
        )

        handler.setFormatter(formatter)

        self.logger.addHandler(handler)

        self.logger.info('Starting TWINS Council')

    def configure_server(self, host, port):
        """Configure the server and open the socket."""
        # List with all the connections to the server
        self.connectionList = []

        # Create a new socket for the server
        self.server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        # Allow the server to be fast reinitialized
        self.server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

        # Included to speed up recv()
        self.server.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self.server.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
        self.server.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPIDLE, 1)
        self.server.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPINTVL, 3)
        self.server.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPCNT, 5)

        # Bind the service to an address and port
        self.server.bind((host, port))
        # We need to define the maximum number of connections
        self.server.listen(1024)

        self.poller = select.epoll()
        self.poller.register(
            self.server, select.POLLIN |
            select.POLLPRI |
            select.POLLHUP |
            select.POLLERR |
            select.POLLOUT
        )

        # Map file descriptors to socket objects
        self.fd_to_socket = {
            self.server.fileno(): self.server,
        }

        # Include the server in the list of connections we are going to listen
        self.connectionList = [self.server]

        # List that will have only the IOFSL servers
        self.iofslServers = []

        # Set to control if we have received communication from all I/O nodes
        self.iofslReceivedSet = set()

        # Dictionary to store the window size for each IOFSL
        self.iofslWindowSize = dict()

        # Default window size
        self.defaultlWindowSize = 1000

        self.defaultPreviousPattern = 'FP'

        self.defaultPreviousOperation = 1   # READ

        # Previous valid window size defined for the server
        self.previousWindowSize = dict()

        self.previousPattern = dict()

        self.previousOperation = dict()

    def train_model(self):
        """Train the neural network to detect the access pattern."""
        # Read the data for training
        csv_path = pd.read_csv(self.MODEL_DATA_PATH, index_col=0, sep=',')

        header = csv_path.dtypes
        self.logger.debug('Features: {}'.format(header))

        df = csv_path.values
        self.logger.debug('Dataframe: {}'.format(df.shape))

        x = df[:, 0:df.shape[1] - 1]    # 0 ... < N-1
        y = df[:, df.shape[1] - 1]      # N-1

        # Apply data transformations
        self.logger.debug('TensorFlow - Applying data transformations')

        self.transformPower = PowerTransformer(method='yeo-johnson').fit(x)
        self.logger.debug('TensorFlow - YeoJohnson - lambdas: {}'.format(
            self.transformPower.lambdas_)
        )

        # Make the transformation
        x = self.transformPower.transform(x)

        self.transformScaler = StandardScaler().fit(x)
        self.logger.debug('TensorFlow - Scaling - mean: {}'.format(
            self.transformScaler.mean_)
        )

        # Make the transformation
        x = self.transformScaler.transform(x)

        self.logger.debug('TensorFlow - Binarizer:')

        self.transformBinarizer = LabelBinarizer()
        self.transformBinarizer.fit_transform(y)

        y = self.transformBinarizer.transform(y)

        self.logger.info('TensorFlow - Features: {}'.format(x.shape))
        self.logger.info('TensorFlow - Output: {}'.format(y.shape))

        # Split the dataset for training and testing
        self.logger.info('TensorFlow - Splitting the dataset for train/test')

        x_train, x_test, y_train, y_test = train_test_split(
            x, y,
            test_size=0.3
        )

        # Check if a saved model exists
        if os.path.isfile(self.MODEL_FILE_PATH):
            # Load the model

            self.logger.info('TensorFlow - Loading the Neural Network model')

            self.model = load_model(self.MODEL_FILE_PATH)
        else:
            # Create a new model and train it

            self.logger.info('TensorFlow - Creating the Neural Network model')

            self.model = Sequential()

            self.model.add(Dense(
                units=x_train.shape[1],
                input_dim=x_train.shape[1],
                activation='relu',
                kernel_initializer='normal',
                name='input_agios_metrics'
            ))
            self.model.add(Dense(
                units=x_train.shape[1],
                activation='relu',
                kernel_initializer='normal',
                name='hidden_layer'
            ))
            self.model.add(Dense(
                units=3,
                activation='softmax',
                name='output_layer'
            ))

            self.model.compile(
                optimizer=RMSprop(lr=0.001),
                loss='categorical_crossentropy',
                metrics=[
                    'accuracy'
                ]
            )

            self.logger.debug('TensorFlow - Summary: {}'.format(
                self.model.summary())
            )

            self.logger.info('TensorFlow - Start training')

            self.model.fit(
                x_train, y_train,
                epochs=self.MODEL_TRAINING_EPOCHS,
                verbose=0,
                batch_size=32,
                validation_split=0.2
            )

            # Save the model
            self.model.save(self.MODEL_FILE_PATH)

            self.logger.info('TensorFlow - TRAIN - Evaluating dataset')

            start = datetime.datetime.now()
            score = self.model.evaluate(x_train, y_train, batch_size=32)
            end = datetime.datetime.now()

            diff = end - start
            self.logger.info('TensorFlow - TRAIN - Time: {}ms'.format(
                diff.total_seconds() * 1000)
            )

            for i in range(0, len(self.model.metrics_names)):
                self.logger.info('TensorFlow - TRAIN - {}: {}'.format(
                    self.model.metrics_names[i], score[i])
                )

            score = self.model.predict(x_test, batch_size=1)

            self.logger.info('TensorFlow - TEST - R2: {}'.format(
                r2_score(y_test, score))
            )

            self.logger.info('TensorFlow - TEST - Evaluating dataset')

            start = datetime.datetime.now()
            score = self.model.evaluate(x_test, y_test, batch_size=32)
            end = datetime.datetime.now()

            diff = end - start
            self.logger.info('TensorFlow - TEST - Time: {}ms'.format(
                diff.total_seconds() * 1000)
            )

            for i in range(0, len(self.model.metrics_names)):
                self.logger.info('TensorFlow - TEST - {}: {}'.format(
                    self.model.metrics_names[i], score[i])
                )

            score = self.model.predict(x_test, batch_size=1)

            self.logger.info('TensorFlow - TEST - R2: {}'.format(
                r2_score(y_test, score))
            )

    def get_best_q(self, operation, pattern):
        """Return the action based on the Q value."""
        best = 0

        # Find the best, i.e., maximum value
        for action in self.Q[operation][pattern]:
            best = max(best, self.Q[operation][pattern][action])

        choices = []

        # Get the choices with that value
        for action in self.Q[operation][pattern]:
            if self.Q[operation][pattern][action] == best:
                choices.append(action)

        # Randomly return one of the good choices
        return choices[random.randint(0, len(choices) - 1)]

    def statistics(self):
        """Dump the statistics to the log."""
        # TODO: dump the statistics to a predefined file
        self.logger.debug('N = {}'.format(self.N))
        self.logger.debug('Q = {}'.format(self.Q))

    def initialize_bandits(self, epsilon, alpha):
        """Initialize the armed bandit instances."""
        # Actions are the window sizes (in ms) that we allow TWINS to use
        self.actions = [
            125,
            250,
            500,
            1000,
            2000,
            4000,
            8000
        ]

        self.classes = [
            'FP',
            'SC',
            'SS'
        ]

        self.operations = [
            0,  # WRITE
            1   # READ
        ]

        self.logger.info('Initiaze Armed Bandit')

        # Initialization of the Armed Bandit
        self.epsilon = epsilon
        self.alpha = alpha

        self.Q = {}
        self.N = {}

        for o in self.operations:
            self.Q[o] = {}
            self.N[o] = {}

            for p in self.classes:
                self.Q[o][p] = {}
                self.N[o][p] = {}

                for a in self.actions:
                    self.Q[o][p][a] = 0.0
                    self.N[o][p][a] = 0

        self.logger.info(self.Q)
        self.logger.info(self.N)

        self.logger.info('Initiaze Armed Bandit - COMPLETE')

    def predict_twins_window(self, host, port, observation):
        """Infer the acces pattern, and predict the best window size."""
        # self.logger.debug('[{}:{}] Metrics: {}'.format(
        #    host, port, observation)
        # )

        agios_operation = int(observation[0, 3])
        agios_size = int(observation[0, 7])     # Size is in bytes

        agios_bandwidth = float(observation[0, 9])
        # self.logger.debug('[{}:{}] Bandwidth: {}'.format(
        #    host, port, agios_bandwidth)
        # )

        # Check if we have requests coming, otherwise report our previous window
        if agios_size == 0:
            if (host, port) in self.previousWindowSize:
                # Get the previous window size for that server
                previous_window = self.previousWindowSize[(host, port)]

                self.logger.warning('[{}] No requests! Using previous window of {}'.format(host, previous_window))
            else:
                # Otherwise get the default window size as no previous was defined
                previous_window = self.defaultlWindowSize

                self.logger.warning('[{}] No requests! Using default window of {}'.format(host, previous_window))

            window = previous_window
        else:
            # We need to organize the data so that we pass only what is needed
            #  4 file_handles
            #  5 avg_received_size
            #  6 min_request_size
            #  7 max_request_size
            #  8 avg_distance
            observation = observation[:, [4, 5, 6, 7, 8]]

            # We need to convert the metrics to be in KB to give to the model
            observation[0][1] = observation[0][1] / 1024.0
            observation[0][2] = observation[0][2] / 1024.0
            observation[0][3] = observation[0][3] / 1024.0
            observation[0][4] = observation[0][4] / 1024.0

            if self.USE_NEURAL_NETWORK:
                # self.logger.debug('[{}] Transforming data'.format(host))

                x = observation

                # Apply data transformations on observation data
                x = self.transformPower.transform(x)
                x = self.transformScaler.transform(x)

                # self.logger.debug('[{}] Making predictions'.format(host))

                # Predict the access pattern
                x = np.array(np.asarray(x))

                y_prob = self.model.predict(x, batch_size=1)

                # self.logger.debug('Probabilities: {}'.format(y_prob))

                detected_access_pattern = self.transformBinarizer.inverse_transform(y_prob)[0]
            else:
                if observation[0][0] > 1:
                    detected_access_pattern = 'FP'
                else:
                    if observation[0][4] > 807680:
                        detected_access_pattern = 'SC'
                    else:
                        if observation[0][4] <= 0:
                            detected_access_pattern = 'SC'
                        else:
                            if observation[0][4] <= 516198.4:
                                detected_access_pattern = 'SS'
                            else:
                                if observation[0][1] <= 128:
                                    detected_access_pattern = 'SC'
                                else:
                                    detected_access_pattern = 'SS'

            self.logger.debug('Access pattern: {}'.format(detected_access_pattern))

            # Record the previous detected access pattern for this node
            self.previousPattern[(host, port)] = detected_access_pattern

            # Record the previous detected operation for this node
            self.previousOperation[(host, port)] = agios_operation

            # Take the best action
            window = self.get_best_q(agios_operation, detected_access_pattern)

        self.logger.debug('TWINS: {}us'.format(window))

        # The reward should be updated based on the pattern we are detecting now and not on the pattern of the previous iteration (thus we first update the previous value and use it to update)

        # Muilti-Armed Bandit Reward
        if agios_size > 0 and agios_bandwidth > 0:

            # If the last action was defined by the multi-armed bandit we need to record the reward
            reward = agios_bandwidth

            action = self.previousWindowSize[(host, port)]
            previous_pattern = self.previousPattern[(host, port)]
            previous_operation = self.previousOperation[(host, port)]

            # Update our value estimation for this action
            self.N[previous_operation][previous_pattern][action] += 1

            if self.alpha is None:
                # If we do not have an alpha defined we should the Incremental implementation
                self.Q[previous_operation][previous_pattern][action] = self.Q[previous_operation][previous_pattern][action] + (1.0 / float(self.N[previous_operation][previous_pattern][action])) * (reward - self.Q[previous_operation][previous_pattern][action])
            else:
                # If we have an alpha we should use the Nonstationary formula
                self.Q[previous_operation][previous_pattern][action] = self.Q[previous_operation][previous_pattern][action] + float(self.alpha) * (reward - self.Q[previous_operation][previous_pattern][action])

        # Return the TWINS window
        return window

    def council_meeting(self):
        """Call a council meeting to decide on the next TWINS window size."""
        # The decision of exploring or exploting SHOULD be made by the Council, as all the clients must do the same thing

        # Determine if we should EXPLORE or EXPLOIT
        if random.randint(1, 100) <= int(self.epsilon):
            # EXPLORATION: take a random action
            council_window_size = self.actions[random.randint(0, len(self.actions) - 1)]

            self.logger.debug('EXPLORE: {}'.format(council_window_size))
        else:
            # EXPLOITATION: take the best action

            # Compute the global window size taking into account the window predicted for each server
            council_window_size = max(set(self.iofslWindowSize.values()), key=self.iofslWindowSize.values().count)

            self.logger.debug('EXPLOIT: {}'.format(council_window_size))

        self.logger.info('COUNCIL - Best window: {}us'.format(council_window_size))

        # Broadcast the new window to eery IOFSL server
        self.announce_window(council_window_size)

        # Statistics
        self.statistics()

        return council_window_size

    def announce_window(self, window):
        """Message all the connected I/O nodes with the new window size."""
        # Send the message to all the nodes
        for iofsl in self.iofslServers:
            # We need a fixed message size
            window = '{}'.format(window).ljust(8)

            # self.logger.debug('Send >{}<'.format(window))

            # Send the message
            iofsl.sendall(window)

        # Reset the set to keep track of new messages
        self.iofslReceivedSet.clear()

        # Reset the defined values (as some nodes may have disconnected)
        self.iofslWindowSize.clear()

    def run(self):
        """Keep listening for new messages and act accordingly."""
        while 1:
            events = self.poller.poll(1000)  # 1s

            for fd, flag in events:
                # Retrieve the actual socket from its file descriptor
                ready_socket = self.fd_to_socket[fd]

                # Handle inputs
                if flag & (select.POLLIN | select.POLLPRI):
                    if ready_socket is self.server:
                        # Accept the new connection
                        newSocket, (remoteHost, remotePort) = self.server.accept()
                        newSocket.setblocking(0)

                        self.fd_to_socket[newSocket.fileno()] = newSocket

                        self.poller.register(
                            newSocket,
                            select.POLLIN |
                            select.POLLPRI |
                            select.POLLHUP |
                            select.POLLERR
                        )

                        # Give the connection a queue for data to send
                        # Include this new connection in our lists to keep track
                        self.connectionList.append(newSocket)
                        self.iofslServers.append(newSocket)

                        self.previousWindowSize[(remoteHost, remotePort)] = self.defaultlWindowSize
                        self.previousPattern[(remoteHost, remotePort)] = self.defaultPreviousPattern
                        self.previousOperation[(remoteHost, remotePort)] = self.defaultPreviousOperation

                        self.logger.info('[{}:{}] Connected'.format(remoteHost, remotePort))

                        window = '{}'.format(self.defaultlWindowSize).ljust(8)
                        newSocket.sendall(window)
                    else:
                        # Receives a fixed size the message and remove trailling spaces
                        message = ready_socket.recv(128).strip()

                        # Captures the information of our peer
                        host, port = ready_socket.getpeername()

                        self.logger.debug('[{}:{}] Received: {}'.format(host, port, message))

                        # Check if the socket was closed by the client
                        if message == '':
                            self.logger.info('Client asked to close socket')

                            # Closes the socket
                            try:
                                self.poller.unregister(ready_socket)
                                ready_socket.close()

                                self.logger.info('Disconnected')
                            except Exception as e:
                                self.logger.error(e)
                                self.logger.error('Unable to shutdown/close socket')

                            try:
                                # Remove the socket from our lists
                                if ready_socket in self.connectionList:
                                    self.connectionList.remove(ready_socket)

                                # Remove the socket from our lists
                                if ready_socket in self.iofslServers:
                                    self.iofslServers.remove(ready_socket)
                            except Exception as e:
                                self.logger.error(e)
                                self.logger.error('Unable to remove socket')

                        else:
                            # We are receiving the following metrics from AGIOS:
                            #  0 pvfs_servers
                            #  1 clients
                            #  2 processes
                            #  3 operation
                            #  4 file_handles
                            #  5 avg_received_size
                            #  6 min_request_size
                            #  7 max_request_size
                            #  8 avg_distance
                            #  9 bandwidth (feedback)

                            # Convert the message we received into an array of metrics to feed to the dataframe
                            observation = np.array([np.fromstring(message, dtype=float, sep=';')])

                            # Include our client in our set, as we received a message from it
                            self.iofslReceivedSet.add((host, port))

                            self.logger.debug('{}/{} messages to call the council'.format(
                                len(self.iofslReceivedSet),
                                len(self.iofslServers))
                            )

                            window = self.predict_twins_window(host, port, observation)

                            # Save what is the best window size for this IOFSL
                            self.iofslWindowSize[(host, port)] = window

                            # If we have information from all the nodes, we can make a council decision (we should compare if it is greater, as we could have removed some closed communications)
                            if len(self.iofslReceivedSet) >= len(self.iofslServers):

                                # Call the council to decide the best window size for all IOFSL servers
                                council = self.council_meeting()

                                # Save the this window as the new previous valid window, if we have a new selection by the Council
                                for k, v in self.previousWindowSize.iteritems():
                                    self.previousWindowSize[k] = council

                elif flag & select.POLLHUP:
                    # Client hung up
                    # Closes the socket
                    try:
                        self.poller.unregister(ready_socket)
                        ready_socket.close()

                        self.logger.info('Disconnected')
                    except Exception as e:
                        self.logger.error(e)
                        self.logger.error('Unable to shutdown/close socket')

                    try:
                        # Remove the socket from our lists
                        if ready_socket in self.connectionList:
                            self.connectionList.remove(ready_socket)

                        # Remove the socket from our lists
                        if ready_socket in self.iofslServers:
                            self.iofslServers.remove(ready_socket)
                    except Exception as e:
                        self.logger.error(e)
                        self.logger.error('Unable to remove socket')

                elif flag & select.POLLERR:
                    # Closes the socket
                    try:
                        self.poller.unregister(ready_socket)
                        ready_socket.close()

                        self.logger.info('Disconnected')
                    except Exception as e:
                        self.logger.error(e)
                        self.logger.error('Unable to shutdown/close socket')

                    try:
                        # Remove the socket from our lists
                        if ready_socket in self.connectionList:
                            self.connectionList.remove(ready_socket)

                        # Remove the socket from our lists
                        if ready_socket in self.iofslServers:
                            self.iofslServers.remove(ready_socket)
                    except Exception as e:
                        self.logger.error(e)
                        self.logger.error('Unable to remove socket')
