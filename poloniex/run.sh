#!/bin/bash
# run
log_file="/home/roni/Desktop/poloniex/logi.log"
while true ; do
	now=$(date +"%T")
	echo "Alotettiin @ : $now" >> $log_file
	python3 mongoexample.py
done
