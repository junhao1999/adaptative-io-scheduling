"""Run the TWINS server Council."""

import argparse

from TWINServerBandit import TWINServerBandit

# Configure the parser
parser = argparse.ArgumentParser()

parser.add_argument(
    '--debug',
    action='store_true',
    dest='debug',
    help='Run TWINS prediction service in debug mode')

parser.add_argument(
    '--host',
    action='store',
    dest='host',
    default='127.0.0.1',
    help='Define the host for the server to listen')

parser.add_argument(
    '--port',
    action='store',
    dest='port',
    default=31713,
    help='Define the port for the server to listen')

parser.add_argument(
    '--epsilon',
    action='store',
    dest='epsilon',
    required=True,
    help='Armed Bandit: epsilon')

parser.add_argument(
    '--alpha',
    action='store',
    dest='alpha',
    required=False,
    help='Armed Bandit: alpha')

# Parse the arguments
args = parser.parse_args()

# Create the service
server = TWINServerBandit(
    args.debug,
    args.host,
    args.port,
    args.epsilon,
    args.alpha
)

server.run()
