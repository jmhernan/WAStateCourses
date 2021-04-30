/* Change column types for data preprocessing step
rename hsCourses to hsCourses_raw */

ALTER TABLE hsCourses RENAME TO hsCourses_raw;
-- Change Gradelevel to int 
CREATE TABLE pre_hsCourses_seq(
  "ResearchID" TEXT,
  "TermEndDate" TEXT,
  "Term" TEXT,
  "CourseID" TEXT,
  "CourseTitle" TEXT,
  "GradeLevelWhenCourseTaken" INTEGER,
  "StateCourseName" TEXT
);
 
INSERT INTO pre_hsCourses_seq(
    ResearchID,
    TermEndDate,
    Term,
    CourseID,
    CourseTitle,
    GradeLevelWhenCourseTaken,
    StateCourseName
)
SELECT ResearchID,
    TermEndDate,
    Term,
    CourseID,
    CourseTitle,
    GradeLevelWhenCourseTaken,
    StateCourseName

FROM hsCourses_raw;