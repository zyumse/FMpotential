#!/bin/sh

let ncore=24

for i in {0..4}
do
mkdir run$i
sed "s/xxx/$RANDOM/g" test.in > run$i/test.in
cd run$i
cp ../tmp.dat tmp.dat
mpirun -np ${ncore} lmp_mpi -in test.in 
cd .. 
done
