DROP VIEW IF EXISTS nsc_cohort_17;
CREATE VIEW nsc_cohort_17
AS
select *
from postSecDems
where EnrollmentBegin >= '2017-06-01' and EnrollmentBegin <= '2017-12-31' and 
v2year4year = 4 and ResearchID in (
    SELECT DISTINCT ResearchID
    FROM enr_2017cohort 
);