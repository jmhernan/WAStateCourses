#!/usr/bin/env zsh


export PROJECTDIR=$HOME/Documents/eScience/projects/WAStateCourses  
export CODEDIR=$PROJECTDIR/seqcrs
export CCER_DATA_DUMP=$HOME/Documents/eScience/data/CCER_cadrs/cadrs_collaboration_data_update
export DB_PATH=$PROJECTDIR/data/ccer_data.db

rm -f DB_PATH

(echo .seperator |; echo .import $CCER_DATA_DUMP/enrollments.txt enrollment) | sqlite3 $DB_PATH
(echo .seperator |; echo .import $CCER_DATA_DUMP/Dim_Student.txt Dim_Student) | sqlite3 $DB_PATH
(echo .seperator |; echo .import $CCER_DATA_DUMP/Dim_School.txt Dim_School) | sqlite3 $DB_PATH
(echo .seperator |; echo .import $CCER_DATA_DUMP/hsCourses.txt hsCourses) | sqlite3 $DB_PATH
(echo .seperator |; echo .import $CCER_DATA_DUMP/postSecDems.txt postSecDems) | sqlite3 $DB_PATH

sqlite3 $DB_PATH '.read '${CODEDIR}'/data/create_cohort.sql'
sqlite3 $DB_PATH '.read '${CODEDIR}'/data/nsc_coverage.sql'

# WIP Some message about rows imported
sqlite3 -batch $DB_PATH << SQL
SELECT COUNT(*) FROM enrollment;
SELECT COUNT(*) FROM Dim_Student;
SELECT COUNT(*) FROM Dim_School;
SELECT COUNT(*) FROM hsCourses;
SELECT COUNT(*) FROM postSecDems;
SQL
