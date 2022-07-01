-- Load Course History File

/* Working with the large student grade history file
    before loading to python */

/* WIP: 
    Steps:
    1. Group by grade level and Order by grade level 9,10,11,12
    2. Order by courseTitle within grade
*/

SELECT ResearchID, GradeLevelWhenCourseTaken, courseTitle
FROM pre_hsCourses_seq
ORDER BY ResearchID, GradeLevelWhenCourseTaken, courseTitle ASC;
