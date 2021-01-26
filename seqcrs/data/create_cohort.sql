DROP VIEW IF EXISTS enr_2017cohort;

CREATE VIEW enr_2017cohort
AS
SELECT *
FROM enrollment enr
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
