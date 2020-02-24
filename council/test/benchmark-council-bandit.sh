#!/bin/bash

echo "TWINS COUNCIL MicroBenchmark"

processes=(32 64 128 256 512 1024 2048)

for n in "${processes[@]}"; do
	echo "benchmarking ${n} processes...";
	mpirun --oversubscribe --np ${n} ./benchmark-council-bandit 127.0.0.1 31713 > "benchmark-${n}-processes-council.csv";
done