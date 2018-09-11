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
