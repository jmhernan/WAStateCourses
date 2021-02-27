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

-- Create Tukwila Cohort
/* To use with Tukwila testing given their manageable enrollment size and CADR completion */
DROP VIEW IF EXISTS enr_2017cohort_tukwila;

CREATE VIEW enr_2017cohort_tukwila
AS
SELECT *
FROM enrollments enr
JOIN Dim_School sch
    ON enr.SchoolCode = sch.SchoolCode
    AND enr.ReportSchoolYear = sch.AcademicYear
WHERE  enr.GradeLevelSortOrder = 15 AND enr.GradReqYear = 2017 AND enr.dGraduate = 1 AND sch.dRoadMapRegionFlag = 1
    AND enr.DistrictCode = 17406;

DROP VIEW IF EXISTS ghf_tukwila;
CREATE VIEW ghf_tukwila
AS
SELECT *
FROM hsCourses 
WHERE ResearchID IN (
SELECT DISTINCT ResearchID
FROM enr_2017cohort_tukwila);

/* Create view of complete SGH records that is atleast one course in 
all 4 years */

-- Check folks that have courses all throughout their HS years 
DROP VIEW IF EXISTS  complete_hs_records;
CREATE VIEW complete_hs_records
AS
SELECT
	ResearchID,
	COUNT(DISTINCT GradeLevelWhenCourseTaken) AS DistinctGradeLevelCount
FROM ghf_cohort_17
WHERE 
	GradeLevelWhenCourseTaken IN (9, 10, 11, 12)
GROUP BY ResearchID;


SELECT COUNT(*) 
FROM ghf_tukwila
WHERE ResearchID IN (SELECT ResearchID FROM complete_hs_records WHERE DistinctGradeLevelCount = 4);
