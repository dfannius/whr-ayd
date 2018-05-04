import os
import sys
import urllib.request

dir = "seasons"
if not os.path.exists(dir):
    os.mkdir(dir)

if len(sys.argv) > 1:
    seasons = [int(sys.argv[1])]
else:
    seasons = range(8, 22)

for season in seasons:
    season_url = "https://ayd.yunguseng.com/season{}/overview.html".format(season)
    season_file = "{}/season_{:02d}.html".format(dir, season)
    print("{} -> {}".format(season_url, season_file))
    urllib.request.urlretrieve(season_url, season_file)
