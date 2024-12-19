## external API calls
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt


import asyncio
import httpx

# work around for Can't invoke asyncio event_loop after tornado 5.0 update #3397
# https://github.com/jupyter/notebook/issues/3397
import nest_asyncio
nest_asyncio.apply()


from _secrets.good_secrets import DD_API_TOKEN, DD_API_URL_BASE
from modules.db_operations import db_client


## External API urls
sic_list_url = f"https://{DD_API_URL_BASE}/DDapi/v1/index_sics/"
sic_data_url = lambda x: f"https://{DD_API_URL_BASE}/DDapi/v1/index_sics/{x.upper()}/prices/"
sic_details_url = lambda x: f"https://{DD_API_URL_BASE}/DDapi/v1/index_sics/{x.upper()}/"

# httpx get Asynchronous wrapper
async def get_url(url, headers ={}, params = {}):
    async with httpx.AsyncClient() as client:
        resp = await client.get(url=url, headers =headers, params = params, timeout=10)
    return resp



# get a list of all SICs
def list_sics(if_full_results: bool = False):
    url = sic_list_url
    headers = {"X-API-KEY": DD_API_TOKEN}
    resp = asyncio.run(get_url(url, headers))
    if resp.status_code == 200:
        res = resp.json()
        if if_full_results:
            res = res
        else:
            res = [x.get("stable_index_code") for x in res]
    else:
        res= []
    return res


# get SIC data to df
def sic2df(sic: str, use_cache: bool = False):
    if use_cache: # using TE's own cached results
        db_c = db_client()
        sic_data = db_c['sic_cache'][sic].find_one({"sic": sic}, {'_id': 0}, sort=[("last_cache_date", -1)])
        if sic_data is not None:
            resp_json = sic_data['sic_data']
        else:
            data_df = pd.DataFrame()
            resp_json = {}
            return data_df, resp_json
    else:
        url = sic_data_url(sic)
        headers = {"X-API-KEY": DD_API_TOKEN}
        params = {"frequency":"N"}
        resp = asyncio.run(get_url(url, headers, params))
        if resp.status_code == 200:
            resp_json = resp.json()
        else:
            data_df = pd.DataFrame()
            resp_json = {}
            return data_df.sort_index(ascending=True, inplace= False), resp_json
    # convert to df
    data_df = pd.DataFrame([x['price'] for x in resp_json],
                               index=[datetime.strptime(x['stable_assigned_date'], "%Y-%m-%d") for x in resp_json],
                               columns=[sic.upper()])
    return data_df.sort_index(ascending=True, inplace= False), resp_json


# get multiple SIC data to one df
def sics2df(sics: list[str], use_cache: bool = False):
    # obtain the sic data for each sic
    sic_dfs = []
    for sic in sics:
        sic_df, _ = sic2df(sic, use_cache)
        if sic_df.size > 0:
            sic_dfs.append(sic_df)
    # merge the data
    if len(sic_dfs) > 0:
        data_df = pd.concat(sic_dfs, axis=1, join="outer")
    else:
        data_df = pd.DataFrame()
    return data_df.sort_index(ascending=True, inplace= False)



# get SIC details to dict
def sic_details(sic: str, use_cache: bool = False):
    if use_cache: # using TE's own cached results
        db_c = db_client()
        res = db_c['sic_cache'][sic].find_one({"sic": sic}, {'_id': 0}, sort=[("last_cache_date", -1)])
        if res is not None:
            res = res['sic_details']
        else:
            res = {}
    else:
        url = sic_details_url(sic)
        headers = {"X-API-KEY": DD_API_TOKEN}
        resp = asyncio.run(get_url(url, headers))
        if resp.status_code == 200:
            res = resp.json()
        else:
            res= {}
    return res

if __name__ == "__main__":


    all_sics = list_sics()
    print(all_sics)
    '''
    df = sics2df(["us98", "us511"])
    # try sic2df
    sic_df, _ = sic2df("us98")
    sic_df.plot(grid = True, figsize = (7,3))
    plt.show()
    # try sic_details
    sic_details("us511")
    '''
