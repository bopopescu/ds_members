WITH by_sku AS (SELECT CASE
	                       WHEN ri.id IS NULL OR ri.reception_status = 'expired' THEN 1
	                       ELSE 0 END   AS kept,
                       b.id             AS box_id,
                       b.kid_profile_id AS kid_id,
                       b.approved_at :: date,
                       v.sku,
                       o.payment_total
                FROM boxes b
	                     JOIN spree_orders o ON b.order_id = o.id
	                     JOIN spree_line_items si ON si.order_id = o.id
	                     LEFT JOIN spree_inventory_units iu ON iu.line_item_id = si.id
	                     LEFT JOIN spree_return_items ri ON ri.inventory_unit_id = iu.id
	                     LEFT JOIN spree_variants v ON v.id = si.variant_id
                WHERE b.state = 'final'
	              AND v.sku <> 'X001-K09-A'),
	 by_box AS (SELECT s.box_id,
	                   s.kid_id,
	                   s.payment_total,
	                   s.approved_at,
	                   sum(kept)                                                      AS items_kept,
	                   rank() OVER (PARTITION BY s.kid_id ORDER BY s.approved_at ASC) AS box_number
	            FROM by_sku s
	            GROUP BY 1, 2, 3, 4),
	 by_kid AS (SELECT b.kid_id,
	                   CASE
		                   WHEN b.box_number > 1 AND items_kept >= 1 AND payment_total >= 0 THEN 1
		                   ELSE 0 END is_good_box
	            FROM by_box b)
SELECT DISTINCT k.kid_id,
                CASE
	                WHEN sum(is_good_box) OVER (PARTITION BY kid_id) > 0 THEN TRUE
	                ELSE FALSE END AS is_good_kid
FROM by_kid k;