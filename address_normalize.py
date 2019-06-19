"""Module to normalize US addresses."""

from postal.parser import parse_address
import pandas as pd
import psycopg2
from sqlalchemy import create_engine
from fuzzywuzzy import process as fwp
from fuzzywuzzy import fuzz
import time
import multiprocessing as mp



redshift = create_engine(
    'postgresql://',
    echo=True,
    pool_recycle=300,
    echo_pool=True,
    creator=lambda _: psycopg2.connect(service='rockets-redshift'))


class address(object):
    """Address normalization class."""

    def __init__(self, address_str=None):
        """Create the address object.
        
        Keyword Arguments:
            address_str {str} -- the address you want to normalize (default: {None})
        """
        self.parsed = parse_address(address_str)
        self.address_dict = {k: v for v, k in self.parsed}

        self.po_box = self.address_dict.get('po_box', None)
        self.house_number = self.address_dict.get('house_number', None)
        self.road = self.address_dict.get('road', None)
        self.level = self.address_dict.get('level', None)
        self.unit = self.address_dict.get('unit', None)
        self.city_district = self.address_dict.get('city_district', None)
        self.city = self.address_dict.get('city', None)
        self.state = self.address_dict.get('state', None)
        self.zipcode = self.address_dict.get('postcode', None)
        self.country = self.address_dict.get('country', None)


    def normalize(self):
        """Custom-made address normalization.
        
        Returns:
            str -- normalized US address

        """
        numb = self.po_box if self.po_box else self.house_number
        road = self.road
        floor = self.level
        unit = self.unit
        city = self.city_district if self.city_district else self.city
        state = self.state
        zipc = self.zipcode[:5] if self.zipcode else None

        norm_add = '{house_number} {road} {floor} {unit} {city} {state} {zipcode}'.format(
            house_number=numb,
            road=road,
            floor=(floor if floor else ''),
            unit=(unit if unit else ''),
            city=city,
            state=(state if state else ''),
            zipcode=(zipc if zipc else ''))
        norm_add = " ".join(norm_add.split())

        return norm_add


    @staticmethod
    def apply_norm(addrs):
        """Normalize an address.
        
        Arguments:
            addrs {str} -- address to normalize
        
        Returns:
            str -- normalized address
        
        """
        a = address(addrs)
        return a.normalize()



def fmatch(test_address, choices, minscore=90):
    """Loose fuzzy address matching, where loose is defined as minimum score=90.
    
    Arguments:
        test_address {str} -- the test address
        choices {list} -- list of all the choices (str) to test against
    
    Returns:
        str -- index of the matched address

    """
    choice, score = fwp.extractOne(test_address, choices)

    return choice if score > minscore else None
    # return choices.index(choice) if score > minscore else None


def cleanup_mult_orders(data):
    """Pick the first row for orders with multiple direct mails.
    
    Arguments:
        data {dataframe} -- all the matched direct mails
    
    Returns:
        dataframe -- a cleaned up version of the dataframe

    """
    singles = data.groupby('order_id').filter(lambda g: len(g) == 1)
    multiples = data.groupby('order_id').filter(lambda g: len(g) > 1)

    cleaned_up = []
    cleaned_up.append(singles)
    for group in multiples.groupby('order_id'):
        cleaned_up.append(group[1].iloc[0,].to_frame().transpose())

    return pd.concat(cleaned_up)


if __name__ == '__main__':
    fdma6 = pd.read_sql_query("""
        SELECT phase,
            test_cell,
            mailing_date,
            left(zipcode, 5) AS zipcode,
            city,
            state,
            address_2 || ' ' || city AS full_address,
            firstname,
            lastname
        FROM dw.fact_direct_mail_addresses
        WHERE phase = 6
            AND test_cell = 2
    """, redshift)

    fdma6['norm_address'] = fdma6['full_address'].apply(address.apply_norm)
    fdma6['short_address'] = fdma6['norm_address'].str[:11].str.cat(fdma6['zipcode'])

    so = pd.read_sql_query(
        """
        SELECT flso.id AS order_id,
            fomp.promotion_code,
            flso.completed_at,
            flso.bill_address1,
            flso.bill_city,
            flso.bill_zipcode,
            flso.bill_address1 || ' ' || flso.bill_city AS order_address,
            flso.ship_firstname,
            flso.ship_lastname,
            flso.bill_firstname,
            flso.bill_lastname
        FROM dw.fact_latest_state_orders flso
        LEFT JOIN dw.fact_order_marketing_promotions fomp
            ON fomp.order_id = flso.id
        WHERE flso.order_type = 'shop'
            AND flso.state = 'complete'
            AND flso.completed_at > DATEADD(HOUR, 12, '2019-05-06')
            AND flso.completed_at < DATEADD(WEEK, 10, '2019-05-06')
            AND bill_address1 IS NOT NULL
    """, redshift)

    so['norm_address'] = so['order_address'].apply(address.apply_norm)
    so['short_address'] = so['norm_address'].str[:11].str.cat(so['bill_zipcode'])

    choices = so['short_address'].tolist()
    pool = mp.Pool(mp.cpu_count())

    # this takes way too long. split into zipcode fuzzy match, then address match
    start_time = time.time()
    fdma6['match_str'] = fdma6['short_address'].apply(fmatch, args=(choices,))
    # fdma6['match_str'] = [
    # results = [
    #     pool.apply(fmatch, args=(row, choices))
    #     for row in fdma6['short_address']
    # ]
    delta_t = time.time() - start_time
    print('matching takes {these} seconds'.format(these=delta_t))

    matchbacks2 = pd.merge(fdma6, so, how='inner', left_on='match_str', right_on='short_address')
    matchbacks2 = matchbacks2.loc[:, [
        'phase', 'test_cell', 'mailing_date', 'zipcode', 'city', 'state',
        'full_address', 'match_str', 'order_id', 'promotion_code',
        'completed_at', 'bill_address1', 'bill_city', 'bill_zipcode',
        'ship_firstname', 'ship_lastname', 'bill_firstname', 'bill_lastname',
        'firstname', 'lastname'
    ]]

    df2 = cleanup_mult_orders(matchbacks2)