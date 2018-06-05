import argparse
import os
import sys
import urllib.request

parser = argparse.ArgumentParser(description="Download YD season files")
parser.add_argument("--league", type=str, default="ayd", metavar="S",
                    help="League (ayd or eyd)")
parser.add_argument("--season", type=int, default="0", metavar="N",
                    help="Single season to download")
args = parser.parse_args()

dir = "{}-seasons".format(args.league)
if not os.path.exists(dir):
    os.mkdir(dir)

if args.season > 0:
    seasons = [args.season]
else:
    start_date = 1 if args.league == "eyd" else 8
    seasons = range(start_date, 22)

for season in seasons:
    season_url = "https://{}.yunguseng.com/season{}/overview.html".format(args.league, season)
    season_file = "{}/season_{:02d}.html".format(dir, season)
    print("{} -> {}".format(season_url, season_file))
    urllib.request.urlretrieve(season_url, season_file)
