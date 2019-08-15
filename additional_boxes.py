import pandas as pd
from redshift import REDSHIFT

d = pd.read_sql_query("""
    WITH all_of_them AS (
        SELECT fb.user_id,
            fb.box_id,
            fb.finalized_at,
            dm.email,
            p.box_id,
            p.timestamp::timestamp                                            AS event_ts,
            CASE WHEN event_ts IS NULL THEN 0 ELSE 1 END                      AS event_fired,
            json_extract_path_text(d.datafields__transactionaldata, 'box_id') AS iterable_box_id,
            CASE WHEN iterable_box_id IS NULL THEN 0 ELSE 1 END               AS email_sent,
            d.email                                                           AS iterable_email,
            d.datafields__createdat,
            d.datafields__workflowname
        FROM dw.fact_boxes fb
                LEFT JOIN dw.dim_members dm ON fb.user_id = dm.user_id
                LEFT JOIN javascript.payment_captured p
                        ON p.box_id = fb.box_id
                LEFT JOIN stitch_iterable.data d
                        ON d.datafields__campaignid = 614275 AND d.eventname = 'emailSend' AND
                            d.datafields__createdat > fb.finalized_at
                            AND json_extract_path_text(d.datafields__transactionaldata, 'box_id') = fb.box_id

        WHERE fb.state = 'final'
        AND fb.finalized_at >= '2019-07-01'
        AND fb.finalized_at < '2019-07-31')

    SELECT DISTINCT email, user_id,
                    SUM(event_fired)  AS events_fired,
                    SUM(email_sent)   AS emails_sent,
                    max(finalized_at) AS ts

    FROM all_of_them
    GROUP BY 1, 2
""", REDSHIFT)

fired = d['events_fired'].map(lambda x: x > 0)
sent = d['emails_sent'].map(lambda x: x > 0)

n_tot = d.shape[0]

not_fired = d.loc[~fired, ['user_id', 'email']]
not_sent = d.loc[fired & ~sent, ['user_id', 'email']]

nf = pd.read_sql_query("""
    SELECT user_id,
        email,
        became_member_at,
        deactivated_at,
        updated_at,
        active,
        state
    FROM dw.fact_subscription_users
    WHERE user_id IN ({nf})
""".format(nf=', '.join([str(x) for x in not_fired['user_id'].unique()])), REDSHIFT)

# why didn't we fire 215 events? (6.5%)
nf.groupby('state')['user_id'].count()

ns = pd.read_sql_query("""
    SELECT user_id,
        email,
        became_member_at,
        deactivated_at,
        updated_at,
        active,
        state
    FROM dw.fact_subscription_users
    WHERE user_id IN ({ns})
""".format(ns=', '.join([str(x) for x in not_sent['user_id'].unique()])), REDSHIFT)

ns.groupby('state')['user_id'].count()
