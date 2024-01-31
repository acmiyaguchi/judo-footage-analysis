#!/usr/bin/env bash
# https://github.com/HumanSignal/label-studio/blob/master/scripts/serve_local_files.sh
# Modified to use nginx and to reverse proxy label-studio
INPUT_DIR=$1
WILDCARD=${2}
OUTPUT_FILE=${3:-"files.txt"}
PORT="8080"

echo "Usage: sh serve_local_files.sh INPUT_DIR WILDCARD OUTPUT_FILE PORT"
echo "This script scans INPUT_DIR directory with WILDCARD filter [all files by default],"
echo "generates OUTPUT_FILE [files.txt by default] with a file list,"
echo "starts web server on the port PORT [8081 by default] that serves files from INPUT_DIR"
echo

# Get the absolute path of the script's parent
SCRIPT_DIR="$(realpath $(dirname ${BASH_SOURCE[0]}))"

echo "Scanning ${INPUT_DIR} ..."
FIND_CMD="find ${INPUT_DIR} -type f"
if [ -z "$WILDCARD" ]; then
  echo "Files wildcard is not set. Serve all files in ${INPUT_DIR}..."
else
  FIND_CMD="${FIND_CMD} -wholename ${WILDCARD}"
fi

echo "Replacing ${INPUT_DIR} to http://localhost:${PORT} ..."
INPUT_DIR_ESCAPED=$(printf '%s\n' "$INPUT_DIR" | sed -e 's/[\/&]/\\&/g')
eval $FIND_CMD | sed "/${INPUT_DIR_ESCAPED}/s//http:\/\/localhost:${PORT}\/data/" > $OUTPUT_FILE

green=`tput setaf 2`
reset=`tput sgr0`
echo "${green}File list stored in '${OUTPUT_FILE}'. Now import it directly from Label Studio UI${reset}"
echo "Showing the first few rows"
head -n3 "${OUTPUT_FILE}"

echo "Running web server on the port ${PORT}"
cd "$INPUT_DIR"

# get the path relative to this script
nginx -c $(realpath ${SCRIPT_DIR}/../config/label-studio/nginx.conf)
