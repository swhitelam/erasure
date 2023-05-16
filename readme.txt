Codes carry out evolutionary learning of neural-network protocols for memory erasure [ref to be added]

I run by compiling taskfarmer

g++ -Wall -o deschamps taskfarmer_info_erasure.c -lm -O

and then running

sbatch deschamps.sh

You'll have to change the cluster-specific commands in taskfarmer_info_erasure.c and deschamps.sh to work on your cluster.

The flow of the code info_erasure.c can be understood by starting from the main() function.