import psycopg2
from sqlalchemy import create_engine
import yaml


try:
    with open('/home/ec2-user/Configs/DsQa/databases.yaml') as c:
        CONF = yaml.full_load(c)
    R = CONF['rockets-redshift']
    S = CONF['rockets-redshift-su']

    REDSHIFT = create_engine(
        'postgresql://',
        echo=False,
        pool_recycle=300,
        echo_pool=True,
        creator=lambda _: psycopg2.connect(
            dbname=R['dbname'],
            user=R['user'],
            password=R['password'],
            host=R['dbhost'],
            port=R['port']
        )
    )

    REDSHIFT_SU = create_engine(
        'postgresql://',
        echo=False,
        pool_recycle=300,
        echo_pool=True,
        creator=lambda _: psycopg2.connect(
            dbname=S['dbname'],
            user=S['user'],
            password=S['password'],
            host=S['dbhost'],
            port=S['port']
        )
    )

    SEGMENT_PROD = CONF['segment']['prod']
    SEGMENT_STG = CONF['segment']['staging']

except:
    REDSHIFT = create_engine(
        'postgresql://',
        echo=True,
        pool_recycle=300,
        echo_pool=True,
        creator=lambda _: psycopg2.connect(service='rockets-redshift'))

    REDSHIFT_SU = create_engine(
        'postgresql://',
        echo=True,
        pool_recycle=300,
        echo_pool=True,
        creator=lambda _: psycopg2.connect(service='rockets-redshift-su'))

    with open('/Users/stallone/ROA/roa_ds_config_future/segment.yaml') as c:
        CONF = yaml.load(c)
        SEGMENT_PROD = CONF['segment']['prod']
        SEGMENT_STG = CONF['segment']['staging']
