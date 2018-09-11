WITH by_sku AS (SELECT CASE
	                       WHEN ri.id IS NULL OR ri.reception_status = 'expired' THEN 1
	                       ELSE 0 END AS kept, b.id AS box_id, b.kid_profile_id AS kid_id, b.season_id, o.user_id, v.sku
                FROM boxes b
	                     JOIN spree_orders o ON b.order_id = o.id
	                     JOIN spree_line_items si ON si.order_id = o.id
	                     LEFT JOIN spree_inventory_units iu ON iu.line_item_id = si.id
	                     LEFT JOIN spree_return_items ri ON ri.inventory_unit_id = iu.id
	                     LEFT JOIN spree_variants v ON v.id = si.variant_id
                WHERE b.state = 'final'
	              AND v.sku <> 'X001-K09-A'
	              AND b.approved_at < (CURRENT_DATE - INTERVAL '2 weeks') :: date
	              AND o.email NOT ILIKE '%@rocketsofawesome.com'),
	 by_box AS (SELECT kid_id, box_id, season_id, (cast(sum(kept) AS float) / cast(count(sku) AS float)) AS keep_rate
	            FROM by_sku
	            GROUP BY kid_id, box_id, season_id)
SELECT kid_id,
       avg(keep_rate) AS avg_keep_rate,
       count(box_id)  AS num_boxes,
	   min(season_id) as first_box_season
FROM by_box
GROUP BY kid_id;