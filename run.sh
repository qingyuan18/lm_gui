#!/usr/bin/env bash
if [[ "$1" == "start" ]]; then
  echo "Starting gui playground..."
  nohup streamlit run ./app_streamlit.py &
elif [[ "$1" == "stop" ]]; then
  echo "Stopping gui playground..."
  ps -ef|grep -v "grep"|grep -i streamlit|awk '{print $2}'|while read line
  do
     kill -9 ${line}
  done
else
   echo "Error: Invalid argument. Usage: ./run.sh start|stop"
  exit 1
fi



