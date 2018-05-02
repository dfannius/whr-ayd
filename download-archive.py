import urllib.request

seasons = range(8, 22)
for season in seasons:
    season_url = "https://ayd.yunguseng.com/season{}/overview.html".format(season)
    season_file = "season_{:02d}.html".format(season)
    print(season_url)
    urllib.request.urlretrieve(season_url, season_file)
