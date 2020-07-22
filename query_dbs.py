import pandas as pd
import psycopg2

subordinate = psycopg2.connect(service="rockets-subordinate")

# All kids with a zip code.
# Even not good customer kids.
# One row per users
kids_and_zip = """
    WITH nkids AS (SELECT user_id,
                          sum(
                              CASE WHEN gender :: text = 'boys' :: text THEN 1 ELSE 0 END) AS  num_boys,
                          sum(
                              CASE WHEN gender :: text = 'girls' :: text THEN 1 ELSE 0 END) AS num_girls,
                          count(*) AS                                                          num_kids
                   FROM kid_profiles
                   GROUP BY user_id
                   ORDER BY user_id)
    SELECT u.id AS user_id,
           u.email,
           u.state,
           CASE WHEN became_member_at IS NOT NULL THEN TRUE ELSE FALSE END AS is_member,
           u.created_at,
           u.became_member_at,
           u.service_fee,
           u.service_fee_enabled,
           a.zipcode,
           nkids.num_boys,
           nkids.num_girls,
           nkids.num_kids
    FROM users u
             JOIN spree_addresses a ON u.ship_address_id = a.id
             JOIN nkids ON nkids.user_id = u.id
    WHERE u.email NOT ILIKE '%@rocketsofawesome.com';
"""

# Average keep rate per user.
# Also number of boxes received per user.
avg_keep_rates = """
    WITH by_sku AS (SELECT CASE
                               WHEN ri.id IS NULL OR ri.reception_status = 'expired' THEN 1
                               ELSE 0 END AS kept, b.id AS box_id, o.user_id, v.sku
                    FROM boxes b
                             JOIN spree_orders o ON b.order_id = o.id
                             JOIN spree_line_items si ON si.order_id = o.id
                             LEFT JOIN spree_inventory_units iu ON iu.line_item_id = si.id
                             LEFT JOIN spree_return_items ri ON ri.inventory_unit_id = iu.id
                             LEFT JOIN spree_variants v ON v.id = si.variant_id
                    WHERE b.state = 'final'  -- otherwise kept is meaningless
                      AND v.sku <> 'X001-K09-A'
                      AND b.approved_at < (CURRENT_DATE - INTERVAL '2 weeks') :: date
                      AND o.email NOT ILIKE '%@rocketsofawesome.com'),
         by_box AS (SELECT user_id, box_id, (cast(sum(kept) AS float) / cast(count(sku) AS float)) AS keep_rate
                    FROM by_sku
                    GROUP BY user_id, box_id)
    SELECT user_id, avg(keep_rate) AS avg_keep_rate, count(box_id) as num_boxes
    FROM by_box
    GROUP BY user_id;
"""

# A good customer must have:
# received a second box
# kept at least one item from that second box
# payment > 0 in that second box
good_customers = """
    WITH skus AS (SELECT CASE
                             WHEN ri.id IS NULL OR ri.reception_status = 'expired' THEN 1
                             ELSE 0 END AS kept,
                         b.id           AS box_id,
                         o.user_id,
                         b.approved_at :: date,
                         v.sku,
                         b.season_id,
                         o.payment_total
                  FROM boxes b
                           JOIN spree_orders o ON b.order_id = o.id
                           JOIN spree_line_items si ON si.order_id = o.id
                           LEFT JOIN spree_inventory_units iu ON iu.line_item_id = si.id
                           LEFT JOIN spree_return_items ri ON ri.inventory_unit_id = iu.id
                           LEFT JOIN spree_variants v ON v.id = si.variant_id
                  WHERE b.state = 'final'
                    AND v.sku <> 'X001-K09-A'),
         box AS (SELECT s.box_id,
                        s.user_id,
                        s.approved_at,
                        s.season_id,
                        s.payment_total,
                        sum(kept) AS items_kept
                 FROM skus s
                 GROUP BY 1, 2, 3, 4, 5),
         by_box AS (SELECT u.id                                                           AS user_id,
                          b.payment_total,
                          rank() OVER (PARTITION BY u.id ORDER BY b.approved_at ASC)     AS box_num,
                          b.items_kept
                   FROM users u
                            INNER JOIN box b ON u.id = b.user_id
                   WHERE u.state = 'subscription_member'
                     AND u.deactivated_at IS NULL

                   ORDER BY user_id,
                            created_at),
    by_user AS (
    SELECT by_box.*,
           CASE
               WHEN box_num > 1 AND items_kept >= 1 AND payment_total >= 0 THEN 1
               ELSE 0 END AS good_box
    FROM by_box)
    SELECT distinct user_id,
           case when sum(good_box) over (PARTITION BY user_id) > 0 then true else false end as is_good_customer
    FROM by_user;
"""

referrals = """
    SELECT DISTINCT u.id AS user_id, CASE WHEN r.claimed IS TRUE THEN TRUE ELSE FALSE END AS is_referred
    FROM users u
             LEFT JOIN referrals r ON r.referred_email = u.email;
"""

first_clicks = """
    SELECT user_id :: int, lower(first_click_source :: text) AS first_click_source
    FROM members.first_click_attribution;
"""

user_agents = """
    SELECT id :: int as user_id,
           context_user_agent
    FROM javascript.users
    WHERE id ~ '^\d+$'
        AND context_user_agent IS NOT NULL;
"""


def query_kid_preferences(kids_list):

    q = pd.read_sql_query(
        """
        SELECT p.id                                                                    AS kid_id,
       CASE
	       WHEN 'black' = ANY(hates_array) THEN 'hates'
	       WHEN 'black' = ANY(loves_array) THEN 'loves'
	       ELSE NULL END                                                       AS black,
       CASE
	       WHEN 'blue' = ANY(hates_array) THEN 'hates'
	       WHEN 'blue' = ANY(loves_array) THEN 'loves'
	       ELSE NULL END                                                       AS blue,
       CASE
	       WHEN 'green' = ANY(hates_array) THEN 'hates'
	       WHEN 'green' = ANY(loves_array) THEN 'loves'
	       ELSE NULL END                                                       AS green,
       CASE
	       WHEN 'orange' = ANY(hates_array) THEN 'hates'
	       WHEN 'orange' = ANY(loves_array) THEN 'loves'
	       ELSE NULL END                                                       AS orange,
       CASE
	       WHEN 'pink' = ANY(hates_array) THEN 'hates'
	       WHEN 'pink' = ANY(loves_array) THEN 'loves'
	       ELSE NULL END                                                       AS pink,
       CASE
	       WHEN 'purple' = ANY(hates_array) THEN 'hates'
	       WHEN 'purple' = ANY(loves_array) THEN 'loves'
	       ELSE NULL END                                                       AS purple,
       CASE
	       WHEN 'red' = ANY(hates_array) THEN 'hates'
	       WHEN 'red' = ANY(loves_array) THEN 'loves'
	       ELSE NULL END                                                       AS red,
       CASE
	       WHEN 'silver' = ANY(hates_array) THEN 'hates'
	       WHEN 'silver' = ANY(loves_array) THEN 'loves'
	       ELSE NULL END                                                       AS silver,
       CASE
	       WHEN 'white' = ANY(hates_array) THEN 'hates'
	       WHEN 'white' = ANY(loves_array) THEN 'loves'
	       ELSE NULL END                                                       AS white,
       CASE
	       WHEN 'yellow' = ANY(hates_array) THEN 'hates'
	       WHEN 'yellow' = ANY(loves_array) THEN 'loves'
	       ELSE NULL END                                                       AS yellow,
       CASE WHEN '3/4-length sleeves' = ANY(blacklist) THEN TRUE ELSE NULL END AS bl_34_length_sleeves,
       CASE WHEN 'bows' = ANY(blacklist) THEN TRUE ELSE NULL END               AS bl_bows,
       CASE WHEN 'button-downs' = ANY(blacklist) THEN TRUE ELSE NULL END       AS bl_button_downs,
       CASE WHEN 'button-down-shirt' = ANY(blacklist) THEN TRUE ELSE NULL END  AS bl_button_down_shirt,
       CASE WHEN 'buttons-on-top' = ANY(blacklist) THEN TRUE ELSE NULL END     AS bl_buttons_on_top,
       CASE WHEN 'cuff-pants' = ANY(blacklist) THEN TRUE ELSE NULL END         AS bl_cuff_pants,
       CASE WHEN 'denim' = ANY(blacklist) THEN TRUE ELSE NULL END              AS bl_denim,
       CASE WHEN 'dresses' = ANY(blacklist) THEN TRUE ELSE NULL END            AS bl_dresses,
       CASE
	       WHEN 'glitter' = ANY(blacklist) OR 'glitter/sequins' = ANY(blacklist) THEN TRUE
	       ELSE NULL END                                                       AS bl_glitter,
       CASE WHEN 'ruffles' = ANY(blacklist) THEN TRUE ELSE NULL END            AS bl_ruffles,
       CASE
	       WHEN 'sequins' = ANY(blacklist) OR 'glitter/sequins' = ANY(blacklist) THEN TRUE
	       ELSE NULL END                                                       AS bl_sequins,
       CASE WHEN 'skirts' = ANY(blacklist) THEN TRUE ELSE NULL END             AS bl_skirts,
       CASE WHEN 'sweaters' = ANY(blacklist) THEN TRUE ELSE NULL END           AS bl_sweaters,
       CASE WHEN 'zippers' = ANY(blacklist) THEN TRUE ELSE NULL END            AS bl_zippers,
       CASE WHEN 'basics' = ANY(outfits) THEN TRUE ELSE NULL END               AS of_basics,
       CASE WHEN 'boys-active-1' = ANY(outfits) THEN TRUE ELSE NULL END        AS of_boys_active_1,
       CASE WHEN 'boys-active-2' = ANY(outfits) THEN TRUE ELSE NULL END        AS of_boys_active_2,
       CASE WHEN 'boys-athleisure' = ANY(outfits) THEN TRUE ELSE NULL END      AS of_boys_athleisure,
       CASE WHEN 'boys-essential' = ANY(outfits) THEN TRUE ELSE NULL END       AS of_boys_essential,
       CASE WHEN 'boys-essentials-1' = ANY(outfits) THEN TRUE ELSE NULL END    AS of_boys_essentials_1,
       CASE
	       WHEN 'boys-performance-active' = ANY(outfits) THEN TRUE
	       ELSE NULL END                                                       AS of_boys_performance_active,
       CASE WHEN 'boys-preppy-1' = ANY(outfits) THEN TRUE ELSE NULL END        AS of_boys_preppy_1,
       CASE WHEN 'boys-preppy-2' = ANY(outfits) THEN TRUE ELSE NULL END        AS of_boys_preppy_2,
       CASE WHEN 'boys-preppy-classic' = ANY(outfits) THEN TRUE ELSE NULL END  AS of_boys_preppy_classic,
       CASE WHEN 'boys-preppy-cool' = ANY(outfits) THEN TRUE ELSE NULL END     AS of_boys_preppy_cool,
       CASE WHEN 'boys-trendy' = ANY(outfits) THEN TRUE ELSE NULL END          AS of_boys_trendy,
       CASE WHEN 'boys-trendy-1' = ANY(outfits) THEN TRUE ELSE NULL END        AS of_boys_trendy_1,
       CASE WHEN 'boys-trendy-2' = ANY(outfits) THEN TRUE ELSE NULL END        AS of_boys_trendy_2,
       CASE WHEN 'classic' = ANY(outfits) THEN TRUE ELSE NULL END              AS of_classic,
       CASE WHEN 'cool' = ANY(outfits) THEN TRUE ELSE NULL END                 AS of_cool,
       CASE WHEN 'girls-active' = ANY(outfits) THEN TRUE ELSE NULL END         AS of_girls_active,
       CASE WHEN 'girls-active-1' = ANY(outfits) THEN TRUE ELSE NULL END       AS of_girls_active_1,
       CASE WHEN 'girls-essential' = ANY(outfits) THEN TRUE ELSE NULL END      AS of_girls_essential,
       CASE WHEN 'girls-essentials-1' = ANY(outfits) THEN TRUE ELSE NULL END   AS of_girls_essentials_1,
       CASE WHEN 'girls-girly' = ANY(outfits) THEN TRUE ELSE NULL END          AS of_girls_girly,
       CASE WHEN 'girls-preppy' = ANY(outfits) THEN TRUE ELSE NULL END         AS of_girls_preppy,
       CASE WHEN 'girls-preppy-1' = ANY(outfits) THEN TRUE ELSE NULL END       AS of_girls_preppy_1,
       CASE WHEN 'girls-preppy-2' = ANY(outfits) THEN TRUE ELSE NULL END       AS of_girls_preppy_2,
       CASE WHEN 'girls-trendy-1' = ANY(outfits) THEN TRUE ELSE NULL END       AS of_girls_trendy_1,
       CASE WHEN 'girls-trendy-2' = ANY(outfits) THEN TRUE ELSE NULL END       AS of_girls_trendy_2,
       CASE WHEN 'girls-trendy-awesome' = ANY(outfits) THEN TRUE ELSE NULL END AS of_girls_trendy_awesome,
       CASE WHEN 'girls-trendy-cool' = ANY(outfits) THEN TRUE ELSE NULL END    AS of_girls_trendy_cool,
       CASE WHEN 'sporty' = ANY(outfits) THEN TRUE ELSE NULL END               AS of_sporty,
       preferences->>'notes'                                                   AS note,
       preferences->'size_preferences'->>'tops'                                AS top_size,
       preferences->'size_preferences'->>'bottoms'                             AS bottom_size,
       CASE
	       WHEN length(preferences->'style_preferences'->>'other') > 0 THEN preferences->'style_preferences'->>'other'
	       ELSE NULL END                                                       AS style_other,
       preferences->'pattern_preferences'->>'camo'                             AS camo,
       preferences->'pattern_preferences'->>'dots'                             AS dots,
       preferences->'pattern_preferences'->>'floral'                           AS floral,
       preferences->'pattern_preferences'->>'hearts'                           AS hearts,
       preferences->'pattern_preferences'->>'leopard'                          AS leopard,
       preferences->'pattern_preferences'->>'plaid'                            AS plaid,
       preferences->'pattern_preferences'->>'stars'                            AS stars,
       preferences->'pattern_preferences'->>'stripes'                          AS stripes,
       CASE
	       WHEN preferences->>'neon' = '1' THEN TRUE
	       WHEN preferences->>'neon' = '0' THEN FALSE
	       ELSE NULL END                                                       AS neon,
       CASE
	       WHEN preferences->>'swim' = '1' THEN TRUE
	       WHEN preferences->>'swim' = '0' THEN FALSE
	       ELSE NULL END                                                       AS swim,
       CASE
	       WHEN preferences->>'text_on_clothes' = '1' THEN TRUE
	       WHEN preferences->>'text_on_clothes' = '0' THEN FALSE
	       ELSE NULL END                                                       AS text_on_clothes,
       CASE
	       WHEN preferences->>'backpack' = '1' THEN TRUE
	       WHEN preferences->>'backpack' = '0' THEN FALSE
	       ELSE NULL END                                                       AS backpack,
       CASE
	       WHEN length(preferences->'sports_preferences'->>'teams') > 1 AND
	            preferences->'sports_preferences'->>'teams' != '[""]' AND
	            preferences->'sports_preferences'->>'teams' != '[]' AND
	            preferences->'sports_preferences'->>'teams' != '[" "]'
		         THEN preferences->'sports_preferences'->>'teams'
	       ELSE NULL END                                                       AS teams
FROM kid_profiles p
	     LEFT JOIN (SELECT p.id, ARRAY(
		                             SELECT jsonb_array_elements_text(p.preferences->'color_preferences'->'loves')) AS loves_array
	                FROM kid_profiles p) col_l ON col_l.id = p.id
	     LEFT JOIN (SELECT p.id, ARRAY(
		                             SELECT jsonb_array_elements_text(p.preferences->'color_preferences'->'hates')) AS hates_array
	                FROM kid_profiles p) col_h ON col_h.id = p.id
	     LEFT JOIN (SELECT p.id,
	                       ARRAY(SELECT jsonb_array_elements_text(p.preferences->'blacklist'->'tags')) AS blacklist
	                FROM kid_profiles p) bl ON bl.id = p.id
	     LEFT JOIN (SELECT p.id, ARRAY(SELECT jsonb_array_elements_text(
		                                          p.preferences->'style_preferences'->'sample_outfits')) AS outfits
	                FROM kid_profiles p) outs ON outs.id = p.id
WHERE p.id IN {kids_list};
    """.format(kids_list=kids_list), subordinate)

    return q