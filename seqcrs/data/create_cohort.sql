DROP VIEW IF EXISTS enr_2017cohort;

CREATE VIEW enr_2017cohort
AS
SELECT *
FROM enrollments enr
JOIN Dim_School sch
    ON enr.SchoolCode = sch.SchoolCode
    AND enr.ReportSchoolYear = sch.AcademicYear
WHERE enr.GradeLevelSortOrder = 15 AND enr.GradReqYear = 2017 AND enr.dGraduate = 1 AND sch.dRoadMapRegionFlag = 1;

--select count(*) from enr_2017cohort;

DROP VIEW IF EXISTS ghf_cohort_17;

CREATE VIEW ghf_cohort_17
AS
SELECT *
FROM hsCourses 
WHERE ResearchID IN (
SELECT DISTINCT ResearchID
FROM enr_2017cohort);

-- Create Renton Cohort
/* To use with Renton testing given their manageable enrollment size and CADR completion */
DROP VIEW IF EXISTS enr_2017cohort_renton;

CREATE VIEW enr_2017cohort_renton
AS
SELECT *
FROM enrollments enr
JOIN Dim_School sch
    ON enr.SchoolCode = sch.SchoolCode
    AND enr.ReportSchoolYear = sch.AcademicYear
WHERE  enr.GradeLevelSortOrder = 15 AND enr.GradReqYear = 2017 AND enr.dGraduate = 1 AND sch.dRoadMapRegionFlag = 1
    AND enr.DistrictCode = 17403;

DROP VIEW IF EXISTS ghf_renton;
CREATE VIEW ghf_renton
AS
SELECT *
FROM hsCourses 
WHERE ResearchID IN (
SELECT DISTINCT ResearchID
FROM enr_2017cohort_renton);