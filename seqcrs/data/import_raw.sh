#!/usr/bin/env zsh


export PROJECTDIR=$HOME/Documents/eScience/projects/WAStateCourses  
export CODEDIR=$PROJECTDIR/seqcrs
export CCER_DATA_DUMP=$HOME/Documents/eScience/data/CCER_cadrs/cadrs_collaboration_data_update
export DB_PATH=$PROJECTDIR/data/ccer_data.db

rm -f DB_PATH

(echo .seperator |; echo .import $CCER_DATA_DUMP/enrollments.txt enrollment) | sqlite3 $DB_PATH




