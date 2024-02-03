import argparse
from bs4 import BeautifulSoup
from collections import namedtuple
import os
import re
import sys
import urllib.request

parser = argparse.ArgumentParser(description="Download YD season files")
parser.add_argument("--league", type=str, default="ayd", metavar="S",
                    help="League (ayd or eyd)")
parser.add_argument("--season", type=int, default="0", metavar="N",
                    help="Single season to download")
args = parser.parse_args()

dir = "{}-overviews".format(args.league)
if not os.path.exists(dir):
    os.mkdir(dir)

if args.season > 0:
    seasons = [args.season]
else:
    start_season = 1 if args.league == "eyd" else 8
    seasons = range(start_season, 100)

overview_re = re.compile(r"/season(\d+)/overview([^_]*)(?:_(.*))?.html")

CycleUrl = namedtuple("CycleFile", ["date", "url", "id"])

archive_url = "https://{}.yunguseng.com/archive.html".format(args.league)
cycle_urls = []
with urllib.request.urlopen(archive_url) as response:
    cur_month = ""
    cur_season = 0
    cur_cycle = 0
    html = response.read()
    soup = BeautifulSoup(html, "lxml")
    for a in soup.find_all("a"):
        if a.contents:
            href = a.get("href")
            if href:
                match = re.match(overview_re, href)
                if match and not "League" in str(a.contents[0]):
                    season = int(match.group(1))
                    id = match.group(2)
                    if id is None: month = id = ""
                    month = match.group(3)
                    if season != cur_season:
                        cur_season = season
                        cur_cycle = 0
                        cur_month = month
                    elif month != cur_month:
                        cur_cycle += 1
                        cur_month = month
                    if cur_season in seasons:
                        cycle_urls.append(CycleUrl(cur_season * 4 + cur_cycle, href, id))

for cycle_url in cycle_urls:
    out_fn = "{}/{:03d}-{}-overview.html".format(dir, cycle_url.date, cycle_url.id)
    in_fn = "https://{}.yunguseng.com{}".format(args.league, cycle_url.url)
    print("Downloading {} to {}".format(in_fn, out_fn))
    urllib.request.urlretrieve(in_fn, out_fn)
