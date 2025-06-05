#! /bin/bash


function run {
	date
	CMD=$1
	
	OUTPUT=$(echo $CMD | cut -d. -f1)_asl_vit.txt
	echo === Command: CMD=$CMD OUTPUT=$OUTPUT
	time bash run-asl-vit.sh
 > jiwoo/$OUTPUT 2>&1
	date
	echo
	echo
}

while read a 
do
	run $a
done <<!
asl.yaml
!