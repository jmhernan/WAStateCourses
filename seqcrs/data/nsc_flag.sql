/* Adding the NSC flag to the sequence table */
-- ADD NSC FLAG
DROP VIEW IF EXISTS nsc_val_outcome;
CREATE VIEW nsc_val_outcome
AS
SELECT a.*, b.nsc_4yr
FROM course_seq_table a
LEFT JOIN(
    SELECT ResearchID, 1 AS nsc_4yr
    FROM nsc_cohort_17
) b ON a.ResearchID = b.ResearchID;