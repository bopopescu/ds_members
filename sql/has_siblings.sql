SELECT kid_id,
       gender,
       birthdate,
       age_rank,
       has_siblings,
       CASE WHEN has_siblings = TRUE AND age_rank > 1 THEN TRUE ELSE FALSE END AS has_older_siblings
FROM (SELECT p.id                                                  AS kid_id,
             u.id                                                  AS user_id,
             p.gender,
             p.birthdate,
             rank() OVER (PARTITION BY user_id ORDER BY birthdate) AS age_rank,
             CASE
	             WHEN (count(*) OVER (PARTITION BY user_id)) > 1 THEN TRUE
	             ELSE FALSE END                                    AS has_siblings
      FROM kid_profiles p
	           INNER JOIN users u ON u.id = p.user_id) k;