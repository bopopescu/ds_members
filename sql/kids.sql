WITH col_l AS (SELECT p.id, ARRAY(
	                            SELECT jsonb_array_elements_text(p.preferences->'color_preferences'->'loves')) AS loves_array
               FROM kid_profiles p),
	 col_h AS (SELECT p.id, ARRAY(
		                        SELECT jsonb_array_elements_text(p.preferences->'color_preferences'->'hates')) AS hates_array
	           FROM kid_profiles p),
	 bl AS (SELECT p.id, ARRAY(SELECT jsonb_array_elements_text(p.preferences->'blacklist'->'tags')) AS blacklist
	        FROM kid_profiles p),
	 outs AS (SELECT p.id, ARRAY(SELECT jsonb_array_elements_text(
		                                    p.preferences->'style_preferences'->'sample_outfits')) AS outfits
	          FROM kid_profiles p)
SELECT p.id AS kid_id,
       CASE
	       WHEN 'black' = ANY(hates_array) THEN 'hates'
	       WHEN 'black' = ANY(loves_array) THEN 'loves'
	       ELSE NULL END                                                        AS black,
       CASE
	       WHEN 'blue' = ANY(hates_array) THEN 'hates'
	       WHEN 'blue' = ANY(loves_array) THEN 'loves'
	       ELSE NULL END                                                        AS blue,
       CASE
	       WHEN 'green' = ANY(hates_array) THEN 'hates'
	       WHEN 'green' = ANY(loves_array) THEN 'loves'
	       ELSE NULL END                                                        AS green,
       CASE
	       WHEN 'orange' = ANY(hates_array) THEN 'hates'
	       WHEN 'orange' = ANY(loves_array) THEN 'loves'
	       ELSE NULL END                                                        AS orange,
       CASE
	       WHEN 'pink' = ANY(hates_array) THEN 'hates'
	       WHEN 'pink' = ANY(loves_array) THEN 'loves'
	       ELSE NULL END                                                        AS pink,
       CASE
	       WHEN 'purple' = ANY(hates_array) THEN 'hates'
	       WHEN 'purple' = ANY(loves_array) THEN 'loves'
	       ELSE NULL END                                                        AS purple,
       CASE
	       WHEN 'red' = ANY(hates_array) THEN 'hates'
	       WHEN 'red' = ANY(loves_array) THEN 'loves'
	       ELSE NULL END                                                        AS red,
       CASE
	       WHEN 'silver' = ANY(hates_array) THEN 'hates'
	       WHEN 'silver' = ANY(loves_array) THEN 'loves'
	       ELSE NULL END                                                        AS silver,
       CASE
	       WHEN 'white' = ANY(hates_array) THEN 'hates'
	       WHEN 'white' = ANY(loves_array) THEN 'loves'
	       ELSE NULL END                                                        AS white,
       CASE
	       WHEN 'yellow' = ANY(hates_array) THEN 'hates'
	       WHEN 'yellow' = ANY(loves_array) THEN 'loves'
	       ELSE NULL END                                                        AS yellow,
       CASE WHEN '3/4-length sleeves' = ANY(blacklist) THEN TRUE ELSE FALSE END AS bl_34_length_sleeves,
       CASE WHEN 'bows' = ANY(blacklist) THEN TRUE ELSE FALSE END               AS bl_bows,
       CASE WHEN 'button-downs' = ANY(blacklist) THEN TRUE ELSE FALSE END       AS bl_button_downs,
       CASE WHEN 'button_down_shirt' = ANY(blacklist) THEN TRUE ELSE FALSE END  AS bl_button_down_shirt,
       CASE WHEN 'buttons_on_top' = ANY(blacklist) THEN TRUE ELSE FALSE END     AS bl_buttons_on_top,
       CASE WHEN 'cuff_pants' = ANY(blacklist) THEN TRUE ELSE FALSE END         AS bl_cuff_pants,
       CASE WHEN 'denim' = ANY(blacklist) THEN TRUE ELSE FALSE END              AS bl_denim,
       CASE WHEN 'dresses' = ANY(blacklist) THEN TRUE ELSE FALSE END            AS bl_dresses,
       CASE
	       WHEN 'glitter' = ANY(blacklist) OR 'glitter/sequins' = ANY(blacklist) THEN TRUE
	       ELSE FALSE END                                                       AS bl_glitter,
       CASE WHEN 'ruffles' = ANY(blacklist) THEN TRUE ELSE FALSE END            AS bl_ruffles,
       CASE
	       WHEN 'sequins' = ANY(blacklist) OR 'glitter/sequins' = ANY(blacklist) THEN TRUE
	       ELSE FALSE END                                                       AS bl_sequins,
       CASE WHEN 'skirts' = ANY(blacklist) THEN TRUE ELSE FALSE END             AS bl_skirts,
       CASE WHEN 'sweaters' = ANY(blacklist) THEN TRUE ELSE FALSE END           AS bl_sweaters,
       CASE WHEN 'zippers' = ANY(blacklist) THEN TRUE ELSE FALSE END            AS bl_zippers,
       CASE WHEN 'basics' = ANY(outfits) THEN TRUE ELSE FALSE END               AS of_basics,
       CASE WHEN 'boys-active-1' = ANY(outfits) THEN TRUE ELSE FALSE END               AS of_boys_active_1,
       CASE WHEN 'boys-active-2' = ANY(outfits) THEN TRUE ELSE FALSE END               AS of_boys_active_2,
       CASE WHEN 'boys-athleisure' = ANY(outfits) THEN TRUE ELSE FALSE END               AS of_boys_athleisure,
       CASE WHEN 'boys-essential' = ANY(outfits) THEN TRUE ELSE FALSE END               AS of_boys_essential,
       CASE WHEN 'boys-essentials-1' = ANY(outfits) THEN TRUE ELSE FALSE END               AS of_boys_essentials_1,
       CASE WHEN 'boys-performance-active' = ANY(outfits) THEN TRUE ELSE FALSE END               AS of_boys_performance_active,
       CASE WHEN 'boys-preppy-1' = ANY(outfits) THEN TRUE ELSE FALSE END               AS of_boys_preppy_1,
       CASE WHEN 'boys-preppy-2' = ANY(outfits) THEN TRUE ELSE FALSE END               AS of_boys_preppy_2,
       CASE WHEN 'boys-preppy-classic' = ANY(outfits) THEN TRUE ELSE FALSE END               AS of_boys_preppy_classic,
       CASE WHEN 'boys-preppy-cool' = ANY(outfits) THEN TRUE ELSE FALSE END               AS of_boys_preppy_cool,
       CASE WHEN 'boys-trendy' = ANY(outfits) THEN TRUE ELSE FALSE END               AS of_boys_trendy,
       CASE WHEN 'boys-trendy-1' = ANY(outfits) THEN TRUE ELSE FALSE END               AS of_boys_trendy_1,
       CASE WHEN 'boys-trendy-2' = ANY(outfits) THEN TRUE ELSE FALSE END               AS of_boys_trendy_2,
       CASE WHEN 'classic' = ANY(outfits) THEN TRUE ELSE FALSE END               AS of_classic,
       CASE WHEN 'cool' = ANY(outfits) THEN TRUE ELSE FALSE END               AS of_cool,
       CASE WHEN 'girls-active' = ANY(outfits) THEN TRUE ELSE FALSE END               AS of_girls_active,
       CASE WHEN 'girls-active-1' = ANY(outfits) THEN TRUE ELSE FALSE END               AS of_girls_active_1,
       CASE WHEN 'girls-essential' = ANY(outfits) THEN TRUE ELSE FALSE END               AS of_girls_essential,
       CASE WHEN 'girls-essentials-1' = ANY(outfits) THEN TRUE ELSE FALSE END               AS of_girls_essentials_1,
       CASE WHEN 'girls-girly' = ANY(outfits) THEN TRUE ELSE FALSE END               AS of_girls_girly,
       CASE WHEN 'girls-preppy' = ANY(outfits) THEN TRUE ELSE FALSE END               AS of_girls_preppy,
       CASE WHEN 'girls-preppy-1' = ANY(outfits) THEN TRUE ELSE FALSE END               AS of_girls_preppy_1,
       CASE WHEN 'girls-preppy-2' = ANY(outfits) THEN TRUE ELSE FALSE END               AS of_girls_preppy_2,
       CASE WHEN 'girls-trendy-1' = ANY(outfits) THEN TRUE ELSE FALSE END               AS of_girls_trendy_1,
       CASE WHEN 'girls-trendy-2' = ANY(outfits) THEN TRUE ELSE FALSE END               AS of_girls_trendy_2,
       CASE WHEN 'girls-trendy-awesome' = ANY(outfits) THEN TRUE ELSE FALSE END               AS of_girls_trendy_awesome,
       CASE WHEN 'girls-trendy-cool' = ANY(outfits) THEN TRUE ELSE FALSE END               AS of_girls_trendy_cool,
       CASE WHEN 'sporty' = ANY(outfits) THEN TRUE ELSE FALSE END               AS of_sporty,
       user_id,
       gender,
       birthdate,
       preferences->>'neon'                                                     AS neon,
       preferences->>'notes'                                                    AS note,
       CASE WHEN length(preferences->>'notes') > 0 THEN TRUE ELSE FALSE END     AS has_note,
       preferences->'size_preferences'->>'tops'                                 AS top_size,
       preferences->'size_preferences'->>'bottoms'                              AS bottom_size,
       preferences->>'text_on_clothes'                                          AS text_on_clothes,
       preferences->'style_preferences'->>'other'                               AS style_other,
       preferences->'pattern_preferences'->>'camo'                              AS camo,
       preferences->'pattern_preferences'->>'dots'                              AS dots,
       preferences->'pattern_preferences'->>'floral'                            AS floral,
       preferences->'pattern_preferences'->>'hearts'                            AS hearts,
       preferences->'pattern_preferences'->>'leopard'                           AS leopard,
       preferences->'pattern_preferences'->>'plaid'                             AS plaid,
       preferences->'pattern_preferences'->>'stars'                             AS stars,
       preferences->'pattern_preferences'->>'stripes'                           AS stripes,
       preferences->>'swim'                                                     AS swim
FROM kid_profiles p
	     LEFT JOIN col_l ON col_l.id = p.id
	     LEFT JOIN col_h ON col_h.id = p.id
	     LEFT JOIN bl ON bl.id = p.id
	     LEFT JOIN outs ON outs.id = p.id;