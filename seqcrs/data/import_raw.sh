#!/usr/bin/env bash


export PROJECTDIR=$HOME/source/WAStateCourses  
export CODEDIR=$PROJECTDIR/seqcrs
export CCER_DATA_DUMP=$HOME/data/cadrs_collaboration_data_update
export DB_PATH=$PROJECTDIR/data/ccer_data.db

rm -f DB_PATH

for file in $CCER_DATA_DUMP/*.txt
    do

        "echo .seperator |"; echo ".import ${file} $(basename ${file} | cut -d. -f1) | sqlite3 $DB_PATH"

    done

sqlite3 $DB_PATH '.read '${CODEDIR}'/data/create_cohort.sql'
sqlite3 $DB_PATH '.read '${CODEDIR}'/data/nsc_coverage.sql'

# WIP Some message about rows imported
sqlite3 -batch $DB_PATH << SQL
SELECT COUNT(*) FROM enrollments;
SELECT COUNT(*) FROM Dim_Student;
SELECT COUNT(*) FROM Dim_School;
SELECT COUNT(*) FROM hsCourses;
SELECT COUNT(*) FROM postSecDems;
SQL
