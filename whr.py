from typing import List, Optional, Tuple

# TODO
# - Predict result of game
# - Show rating changes for players with new games
# - Smarter choice of what player to update

import argparse
from bs4 import BeautifulSoup, NavigableString
import csv
import glob
import math
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
import re
import sys

parser = argparse.ArgumentParser(description="WHR for AYD")
parser.add_argument("--games-file", type=str, default="games.csv", metavar="F",
                    help="File of game data")
parser.add_argument("--ratings-file", type=str, default="ratings.csv", metavar="F",
                    help="File of ratings data")
parser.add_argument("--report-file", type=str, default="report.txt", metavar="F",
                    help="File with ratings report")
parser.add_argument("--read-games", action="store_true", default=False,
                    help="Read games file before parsing new seasons")
parser.add_argument("--parse-seasons", action="store_true", default=False,
                    help="Parse HTML season files into game data file")
parser.add_argument("--parse-season-start", type=int, default=1, metavar="N",
                    help="Season to start parsing HTML files from")
parser.add_argument("--analyze-games", action="store_true", default=False,
                    help="Analyze game data file")
parser.add_argument("--print-report", action="store_true", default=False,
                    help="Produce ratings report")
parser.add_argument("--load-ratings", action="store_true", default=False,
                    help="Load data from ratings file")
parser.add_argument("--draw-graph", type=str, default=None, metavar="H",
                    help="Handle of user's graph to draw")
parser.add_argument("--whr-vs-yd", action="store_true", default=False,
                    help="Draw scatterplot of WHR vs YD ratings")
parser.add_argument("--league", type=str, default="ayd", metavar="S",
                    help="League (ayd or eyd)")
parser.add_argument("--leagues", type=str, default="", metavar="S",
                    help="Leagues")
parser.add_argument("--min-date", type=int, default=0, metavar="N",
                    help="Mininum active date for players in graph")
parser.add_argument("--note-new-games", action="store_true", default=False,
                    help="Print the results of newly parsed games")

args = parser.parse_args()
if len(args.leagues) == 0:
    args.leagues = args.league

first_season = 0                # No longer really used

rating_scale = 1.5
rating_shift = -1.2

# "rank" roughly corresponds to AGA ratings.
def rating_to_rank(raw_r):
    # To convert to AGA ratings it seems that we should divide raw_r
    # by 1.6, but to get ratings to match up at all at both the top
    # and bottom of the population I need to multiply instead.
    return raw_r * rating_scale + rating_shift

def rank_to_rank_str(rank, integral=False):
    if integral and int(rank) == rank:
        rank = int(rank)
        if rank == 1:
            return "1d/1k"
        dan_fmt = "{}d"
        kyu_fmt = "{}k"
    else:
        dan_fmt = "{:.2f}d"
        kyu_fmt = "{:.2f}k"
    if rank >= 1:
        return dan_fmt.format(rank)
    else:
        return kyu_fmt.format(2-rank)

def rating_to_rank_str(raw_r):
    return rank_to_rank_str(rating_to_rank(raw_r))

def r_to_gamma(r):
    return math.exp(r)

# A 'season' consists of three 'cycles' (each lasting one month) and
# then an off month. 'dates' are in units of months. So the date goes
# up by 4 with each new season.

def date_to_season_cycle(d):
    season = int(d/4) + first_season
    cycle = d - (season-first_season) * 4
    return (season, cycle)

def date_to_str(d):
    season, cycle = date_to_season_cycle(d)
    return "{}{}".format(season, "ABC"[cycle])

def season_cycle_to_date(s, c):
    return (s - first_season)*4 + c

class RatingDatum:
    """The rating of a player at a particular date."""
    def __init__(self, date: int, rating: float, std: float=0) -> None:
        self.date: int = date
        self.rating: float = rating
        self.gamma: float = r_to_gamma(self.rating)
        self.std: float = std
        self.wins: List[Player] = []          # Players defeated on this date
        self.losses: List[Player] = []        # Players lost to on this date

    def set_std(self, std: float):
        self.std = std

    def __repr__(self):
        return "{}: {}".format(self.date, rating_to_rank_str(self.rating))

class Player:
    def __init__(self, name, handle, player_db, is_root=False):
        self.name: str = name
        self.handle: str = handle
        self.games: List[Game] = []
        self.yd_ratings: Mapping[int, RatingDatum] = {}
        self.rating_history = []
        self.player_db = player_db
        self.root = is_root
        self.rating_hash = {}

    def __repr__(self):
        return "{} ({})".format(self.name, self.handle)

    def set_yd_rating(self, league: str, rating: float):
        self.yd_ratings[league] = rating

    def set_ayd_rating(self, ayd_rating: float):
        self.set_yd_rating("ayd", ayd_rating)
        
    def set_eyd_rating(self, eyd_rating: float):
        self.set_yd_rating("eyd", eyd_rating)
        
    def get_yd_rating(self, league: str) -> float:
        return self.yd_ratings.get(league, 0)
        
    def get_ayd_rating(self) -> float:
        return self.get_yd_rating("ayd")

    def get_eyd_rating(self) -> float:
        return self.get_yd_rating("eyd")

    def include_in_graph(self) -> bool:
        if self.root: return False
        if len(self.rating_history) <= 1: return False
        if len(self.get_wins()) == 0 or len(self.get_losses()) == 0: return False
        if self.handle == "Chimin": return False
        return True

    def write_rating_history(self, f):
        print('"{}","{}"'.format(self.name, self.handle), file=f, end="")
        print(',{}'.format(self.get_ayd_rating()), file=f, end="")
        print(',{}'.format(self.get_eyd_rating()), file=f, end="")
        print(',{}'.format(len(self.rating_history)), file=f, end="")
        for r in self.rating_history:
            print(',{},{:.8f},{:.8f}'.format(r.date, r.rating, r.std), file=f, end="")
        print(file=f)

    # Takes a CSV row, already split
    def read_rating_history(self, row: List[str]):
        self.set_ayd_rating(int(row[0]))
        self.set_eyd_rating(int(row[1]))

        row = row[2:]
        num_points = int(row[0])
        appending = False
        rating_idx = 0
        for i in range(num_points):
            (date, rating, std) = (int(row[3*i+1]),
                                   float(row[3*i+2]),
                                   float(row[3*i+3]))
            if not appending:
                while (rating_idx < len(self.rating_history) and
                       self.rating_history[rating_idx].date != date):
                    rating_idx += 1
                if rating_idx >= len(self.rating_history):
                    appending = True
                else:
                    r = self.rating_history[rating_idx]
                    r.rating = rating
                    r.gamma = r_to_gamma(rating)
                    r.std = std

            if appending:
                self.rating_history.append(RatingDatum(date, rating, std))

        for r in self.rating_history:
            self.rating_hash[r.date] = r

    def copy_rating_history_from(self, other: "Player"):
        self.yd_ratings = other.yd_ratings.copy()
        for r in self.rating_history:
            if r.date in other.rating_hash:
                other_r = other.rating_hash[r.date]
                r.rating = other_r.rating
                r.gamma = other_r.gamma
                r.std = other_r.std

    def add_game(self, game: "Game"):
        self.games.append(game)

    def remove_recent_games(self, start_date: int):
        self.games = [g for g in self.games if g.date < start_date]

    def latest_rating(self) -> float:
        if len(self.rating_history) == 0:
            return 0
        else:
            return self.rating_history[-1].rating

    def latest_std(self) -> float:
        if len(self.rating_history) == 0:
            return 0
        else:
            return self.rating_history[-1].std

    def get_rating_fast(self, date) -> float:
        if self.root:
            return 0
        else:
            return self.rating_hash[date].rating

    def get_gamma_fast(self, date) -> float:
        if self.root:
            return 1
        else:
            return self.rating_hash[date].gamma

    def get_wins(self) -> List[Tuple[int, float]]:
        results = []
        for r in self.rating_history:
            for w in r.wins:
                if not w.root:
                    results.append((r.date, w.get_rating(r.date)))
        return results

    def get_losses(self) -> List[Tuple[int, float]]:
        results = []
        for r in self.rating_history:
            for l in r.losses:
                if not l.root:
                    results.append((r.date, l.get_rating(r.date)))
        return results

    def get_rating(self, date: int) -> float:
        if self.root:
            return 0
        if len(self.rating_history) == 0:
            return 0
        # Dog-slow implementation for now. In fact, everyone has a rating history point
        # for every game that they have played, so we will only ever hit the first arm
        # of this.
        for (i, r) in enumerate(self.rating_history):
            if r.date == date:
                return r.rating
            elif r.date > date:
                if i == 0:
                    return r.rating
                else:
                    prev_r = self.rating_history[i-1]
                    frac = (date - prev_r.date) / (r.date - prev_r.date)
                    rating = prev_r.rating + (r.rating - prev_r.rating) * frac
                    return rating
        return self.rating_history[-1].rating

    def init_rating_history(self):
        if self.root: return
        min_date = min((g.date for g in self.games), default=0)
        root_date = min_date - 100
        root_player = self.player_db.get_root_player()
        self.add_game(Game(root_date, self, root_player))
        self.add_game(Game(root_date, root_player, self))
        dates = list(set(g.date for g in self.games))
        dates.sort()
        self.games.sort(key=lambda g: g.date)

        self.rating_history = [RatingDatum(d, 0) for d in dates]
        for r in self.rating_history:
            self.rating_hash[r.date] = r
        i = 0
        for g in self.games:
            if self.rating_history[i].date != g.date:
                i += 1
                assert(self.rating_history[i].date == g.date)
            if g.winner == self:
                self.rating_history[i].wins.append(g.loser)
            else:
                self.rating_history[i].losses.append(g.winner)

    # Return (Hessian, gradient)
    def compute_derivatives(self):
        # The WHR paper expresses w^2 in units of Elo^2/day. The conversion to r^2/month
        # means multiplying by (ln(10) / 400)^2 * 30 ~= 0.001
        elo_wsq = 100           # I've also tried 300 but this looks good
        wsq = elo_wsq * 0.001

        num_points = len(self.rating_history)
        H = np.eye(num_points) * -0.001
        g = np.zeros(num_points)
        for (i,r) in enumerate(self.rating_history):
            my_gamma = r.gamma
            g[i] += len(r.wins)                              # Bradley-Terry
            for j in r.wins + r.losses:
                their_gamma = j.get_gamma_fast(r.date)
                factor = 1. / (my_gamma + their_gamma)
                g[i] -= my_gamma * factor                    # Bradley-Terry
                H[i,i] -= my_gamma * their_gamma * factor**2 # Bradley-Terry
            if i > 0:
                dr = r.rating - self.rating_history[i-1].rating
                dt = r.date - self.rating_history[i-1].date
                sigmasq_recip = 1./(dt * wsq)
                g[i] -= dr * sigmasq_recip                   # Wiener
                H[i,i] -= sigmasq_recip                      # Wiener
                if i >= 1:
                    g[i-1] += dr * sigmasq_recip             # Wiener
                    H[i-1,i-1] -= sigmasq_recip              # Wiener
                H[i-1,i] += sigmasq_recip                    # Wiener
                H[i,i-1] += sigmasq_recip                    # Wiener
        return (H, g)

    # Return magnitude of changes
    def iterate_whr(self) -> float:
        if self.root: return 0.0

        (H, g) = self.compute_derivatives()
        num_points = H.shape[0]
        # xn = np.linalg.solve(H, g)       # for double-checking

        d = np.zeros(num_points)
        b = np.zeros(num_points)
        a = np.zeros(num_points)
        d[0] = H[0,0]
        if num_points > 1:
            b[0] = H[0,1]
            for i in range(1, num_points):
                a[i] = H[i,i-1] / d[i-1]
                d[i] = H[i,i] - a[i] * b[i-1]
                if i < num_points-1:
                    b[i] = H[i,i+1]
        
        y = np.zeros(num_points)
        x = np.zeros(num_points)
        y[0] = g[0]
        for i in range(1, num_points):
            y[i] = g[i] - a[i] * y[i-1]
        x[num_points-1] = y[num_points-1] / d[num_points-1]
        for i in range(num_points - 2, -1, -1):
            x[i] = (y[i] - b[i] * x[i+1]) / d[i]

        for (i,r) in enumerate(self.rating_history):
            r.rating -= x[i]
            r.gamma = r_to_gamma(r.rating)

        return np.linalg.norm(x)

    # Return list of std deviation at each rating point
    def compute_stds(self):
        (H, g) = self.compute_derivatives()
        num_points = H.shape[0]
        # We're only doing this once, I don't care about speed tricks
        Hinv = np.linalg.inv(H)
        for (i,r) in enumerate(self.rating_history):
            r.set_std(math.sqrt(-Hinv[i,i]))

class Game:
    def __init__(self, date: int, winner: Player, loser: Player) -> None:
        self.date: int = date
        self.winner: Player = winner
        self.loser: Player = loser

    def __eq__(self, other):
        return (self.date == other.date and
                self.winner == other.winner and
                self.loser == other.loser)

    def __repr__(self):
        return "{:02}: {} > {}".format(self.date, self.winner, self.loser)

class PlayerDB:
    def __init__(self):
        self.player_map: Mapping[str, Player] = {}
        self.root_player: Player = self.get_player("[root]", "[root]", is_root=True)

    def get_player(self, name: str, handle: str, is_root=False) -> Player:
        if handle in self.player_map:
            player = self.player_map[handle]
            assert(player.name == name)
            return player
        else:
            player = Player(name, handle, self, is_root)
            self.player_map[handle] = player
            return player

    def get_root_player(self) -> Player:
        return self.root_player

    def copy_rating_history_from(self, other_db: "PlayerDB"):
        for (handle, player) in self.player_map.items():
            if handle in other_db.player_map:
                player.copy_rating_history_from(other_db.player_map[handle])

    def remove_recent_games(self, start_date: int):
        for p in self.player_map.values():
            p.remove_recent_games(start_date)

    def __getitem__(self, handle: str) -> Player:
        return self.player_map[handle]

    def __len__(self) -> int:
        return len(self.player_map)

    def clear(self):
        self.player_map.clear()

    def values(self) -> List[Player]:
        return self.player_map.values()

the_player_db = PlayerDB()

def is_cycle_name(tag):
    return (tag.name == "b" and tag.contents[0].startswith("AYD") or
            tag.name == "h3" and "League" in tag.contents[0])

# A full cycle name is something like "AYD League B, March 2014".
# Get just the date part
def cycle_to_date(s: str) -> str:
    return s.split(", ")[-1]

def flush_old_games(player_db: PlayerDB, start_season: int, old_games: List[Game]):
    start_date = season_cycle_to_date(start_season, 0)
    player_db.remove_recent_games(start_date)
    games = [g for g in old_games if g.date < start_date]
    flushed_games = [g for g in old_games if g.date >= start_date]
    return games, flushed_games

def parse_seasons(player_db: PlayerDB,
                  league: str,
                  start_season: int,
                  out_fname: str,
                  existing_games: List[Game] ,
                  flushed_games: List[Game]):
    start_date = season_cycle_to_date(start_season, 0)

    # An overview file may contain all three cycles of a season (AYD,
    # early EYD seasons) or a single cycle (late EYD seasons).
    overview_files = glob.glob("{}-overviews/*-overview.html".format(league))
    overview_file_array: List[Optional[str]] = []
    overview_file_re = re.compile(r"(\d+)-overview.html")
    for fn in overview_files:
        match = re.search(overview_file_re, fn)
        if match:
            date = int(match.group(1))
            while date >= len(overview_file_array):
                overview_file_array.append(None)
            overview_file_array[date] = fn

    # player_db.remove_recent_games(start_date)
    # games = [g for g in existing_games if g.date < start_date]
    games = existing_games
    new_games: List[Game] = []

    anchor_date = start_date
    while anchor_date < len(overview_file_array):
        fn = overview_file_array[anchor_date]
        if fn is None:
            anchor_date += 1
            continue
        print("{}...".format(anchor_date), end="", flush=True)
        with open(fn, "rb") as f:
            soup = BeautifulSoup(f, "lxml")

            # First find the names of the cycles
            season_cycles: List[str] = [] # Names of cycles in this season, in chronological order

            cycle_tags = soup.find_all(is_cycle_name)
            for cycle_tag in cycle_tags:
                date_name = cycle_to_date(cycle_tag.contents[0])
                if date_name not in season_cycles:
                    season_cycles.append(date_name)
            season_cycles.reverse()

            # Now find the crosstables
            crosstables = soup.find_all("table", id="pointsTable")
            for crosstable in crosstables:
                # Find the name and date of this table
                crosstable_name = crosstable.find_parent("table").previous_sibling
                while type(crosstable_name) is NavigableString or crosstable_name.name != "h3":
                    crosstable_name = crosstable_name.previous_sibling
                crosstable_date = cycle_to_date(crosstable_name.contents[0])
                date = anchor_date + season_cycles.index(crosstable_date)

                # Construct the list of players, in order
                crosstable_players = []
                trs = crosstable.find_all("tr")
                for tr in trs:
                    tds = tr.find_all("td")
                    if len(tds) > 0:
                        name = tds[1].nobr.a.contents[0]
                        handle = tds[2].contents[0]
                        yd_rating = int(tds[11].contents[0])
                        player = player_db.get_player(name, handle)
                        player.set_yd_rating(league, yd_rating)
                        crosstable_players.append(player)

                # Parse game results
                row_player_idx = 0
                for tr in trs:
                    tds = tr.find_all("td")
                    if len(tds) > 0:
                        for col_player_idx in range(row_player_idx + 1, len(crosstable_players)):
                            gif = tds[3+col_player_idx].find("img")["src"]
                            if gif:
                                if gif.endswith("won.gif"):
                                    winner = crosstable_players[row_player_idx]
                                    loser = crosstable_players[col_player_idx]
                                elif gif.endswith("lost.gif"):
                                    winner = crosstable_players[col_player_idx]
                                    loser = crosstable_players[row_player_idx]
                                else: # empty or forfeit
                                    continue
                                game = Game(date, winner, loser)
                                if game.date >= start_date and game not in flushed_games:
                                    new_games.append(game)
                                games.append(game)
                                winner.add_game(game)
                                loser.add_game(game)
                        row_player_idx += 1
        anchor_date += 1

    with open(out_fname, "w") as out_file:
        for g in games:
            print('{},"{}","{}","{}","{}"'.format(g.date,
                                                  g.winner.name,
                                                  g.winner.handle,
                                                  g.loser.name,
                                                  g.loser.handle),
                  file=out_file)

    if new_games and args.note_new_games:
        print("\nNew games:")
        for g in new_games:
            print("  {} > {}".format(g.winner.handle, g.loser.handle))

# Read in the games file that was produced by analyze_seasons()
# and append it to `games`
def read_games_file(player_db: PlayerDB, fname: str, games: List[Game]) -> List[Game]:
    with open(fname) as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            (date_str, winner_name, winner_handle, loser_name, loser_handle) = row
            date = int(date_str)
            winner = player_db.get_player(winner_name, winner_handle)
            loser = player_db.get_player(loser_name, loser_handle)
            game = Game(date, winner, loser)
            games.append(game)
            winner.add_game(game)
            loser.add_game(game)
    return games

def init_whr(player_db: PlayerDB):
    for p in player_db.values():
        p.init_rating_history()

def iterate_whr(player_db: PlayerDB):
    max_change = -1
    max_change_player = None
    for p in player_db.values():
        change = abs(p.iterate_whr())
        if change > max_change:
            max_change = change
            max_change_player = p
    # print(f"Most changed player: {p.handle} = {p.latest_rating()}")
    return max_change

def run_whr(player_db: PlayerDB):
    for i in range(1000):
        if (i+1) % 100 == 0:
            print("{}...".format(i+1), end="", flush=True)
            # print("ITERATION {}".format(i))
        max_change = iterate_whr(player_db)
        if max_change < 1e-06:
            print("Completed WHR in {} iteration{}...".format(i+1, "s" if i > 0 else ""),
                  end="", flush=True)
            break
    for p in player_db.values():
        p.compute_stds()

def print_report(player_db: PlayerDB, fname: str):
    with open(fname, "w") as f:
        for p in sorted(player_db.values(), key=lambda p: p.latest_rating(), reverse=True):
            if len(p.rating_history) > 0:
                print("{:<10} {:>5} ± {:.2f}: {}".format(p.handle,
                                                         rating_to_rank_str(p.latest_rating()),
                                                         p.rating_history[-1].std,
                                                         p.rating_history[1:]),
                      file=f)

def save_rating_history(player_db: PlayerDB, fname: str):
    with open(fname, "w") as f:
        for p in sorted(player_db.values(), key=lambda p: p.latest_rating(), reverse=True):
              p.write_rating_history(f)

def load_rating_history(player_db: PlayerDB, fname: str):
    with open(fname) as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            (name, handle) = row[:2]
            p = player_db.get_player(name, handle)
            p.read_rating_history(row[2:])

def league_games_file(league: str):
    return "{}-{}".format(league, args.games_file)

ratings_file = "{}-{}".format(args.league, args.ratings_file)
report_file = "{}-{}".format(args.league, args.report_file)

leagues = args.leagues.split(",")

if args.parse_seasons:
    games: List[Game] = []
    if args.read_games:
        the_player_db.clear()
        print("Reading games file...", end="", flush=True) 
        games = []
        for league in leagues:
            print("{}...".format(league), end="")
            games = read_games_file(the_player_db, league_games_file(league), games)
    print("Parsing seasons...", end="", flush=True) 
    for league in leagues:
        print("{}...".format(league), end="")
        games, flushed_games = flush_old_games(the_player_db, args.parse_season_start, games)
        parse_seasons(the_player_db,
                      league,
                      args.parse_season_start,
                      league_games_file(league),
                      games,
                      flushed_games)
else:
    print("Reading games file...", end="", flush=True) 
    the_player_db.clear()
    games = []
    for league in leagues:
        print("{}...".format(league), end="")
        games = read_games_file(the_player_db, league_games_file(league), games)
init_whr(the_player_db)

need_ratings = args.print_report or args.draw_graph or args.whr_vs_yd
if args.load_ratings or (need_ratings and not args.analyze_games):
    old_player_db = PlayerDB()
    load_rating_history(old_player_db, ratings_file)
    the_player_db.copy_rating_history_from(old_player_db)

if args.analyze_games:
    print("Running WHR...", end="", flush=True) 
    run_whr(the_player_db)
    save_rating_history(the_player_db, ratings_file)

if args.print_report:
    print("Printing report...", end="", flush=True) 
    print_report(the_player_db, report_file)

plt.style.use("seaborn-darkgrid")

def date_str_ticks(dates: List[int]):
    # Put a label just on the middle cycle of each season, unless the player didn't play that one.
    tick_labels = ["" for d in dates]
    seasons_seen = set()
    seasons_cycles = [date_to_season_cycle(d) for d in dates]
    # First label middle cycles
    for (i,d) in enumerate(dates):
        (season, cycle) = seasons_cycles[i]
        if cycle == 1:
            tick_labels[i] = str(season)
            seasons_seen.add(season)
    # Make sure we label seasons whose middle cycles are missing
    for (i,d) in enumerate(dates):
        (season, cycle) = seasons_cycles[i]
        if not season in seasons_seen:
            tick_labels[i] = str(season)
            seasons_seen.add(season)

    return tick_labels

if args.draw_graph:
    plot_dir = "plots"
    if not os.path.exists(plot_dir):
        os.mkdir(plot_dir)

    handle = args.draw_graph
    p = the_player_db[handle]
    history = p.rating_history[1:]
    dates = [r.date for r in history]
    ratings = [r.rating for r in history]
    # with plt.rc_context({'axes.autolimit_mode': 'round_numbers'}):

    plt.title("\n" + handle + "\n")
    plt.fill_between(dates,
                     [rating_to_rank(r.rating - r.std) for r in history], 
                     [rating_to_rank(r.rating + r.std) for r in history],
                     alpha=0.2)
    ranks = [rating_to_rank(r) for r in ratings]
    plotted_ranks = ranks
    plt.plot(dates, ranks)

    wins = p.get_wins()
    if len(wins) > 0:
        win_dates, win_ratings = zip(*wins)
        win_ranks = [rating_to_rank(r) for r in win_ratings]
        plotted_ranks += win_ranks
        plt.scatter(win_dates, win_ranks, edgecolors="green", facecolors="none", marker="o")
    losses = p.get_losses()
    if len(losses) > 0:
        loss_dates, loss_ratings = zip(*losses)
        loss_ranks = [rating_to_rank(r) for r in loss_ratings]
        plotted_ranks += loss_ranks
        plt.scatter(loss_dates, loss_ranks, color="red", marker="x")
    
    y_min = int(min(plotted_ranks) - 1)
    y_max = int(max(plotted_ranks) + 1)
    plt.ylim(y_min, y_max)

    # (tick_vals, tick_labels) = plt.yticks()
    new_tick_vals = np.arange(y_min, y_max + 1, 1.0)
    new_tick_labels = [rank_to_rank_str(r, True) for r in new_tick_vals]
    plt.yticks(new_tick_vals, new_tick_labels)
    plt.xticks(dates, date_str_ticks(dates))
    plt.xlabel("Season")
    plt.ylabel("Rank")
    plt.savefig("{}/{}.png".format(plot_dir, handle))
    plt.show()

if args.whr_vs_yd:
    players = [p for p in the_player_db.values() if p.include_in_graph()]
    players = [p for p in players if p.rating_history[-1].date >= args.min_date]
    whr_ranks = [rating_to_rank(p.latest_rating()) for p in players]
    whr_stds = [rating_scale * p.latest_std() for p in players]
    yd_ratings = [p.get_yd_rating(args.league) for p in players]

    W = np.vstack([whr_ranks, np.ones(len(whr_ranks))]).T
    (lsq, resid, rank, sing) = np.linalg.lstsq(W, yd_ratings, rcond=None)

    fig, ax = plt.subplots()
    ax.scatter(whr_ranks, yd_ratings, s=4)
    ax.errorbar(whr_ranks, yd_ratings, xerr=whr_stds, fmt="none", linewidth=0.2)
    ax.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(base=1))
    ax.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(base=100))
    tick_vals = ax.get_xticks()
    tick_labels = ax.get_xticklabels()
    new_tick_labels = [rank_to_rank_str(r, True) for r in tick_vals]
    ax.set_xticks(tick_vals)
    ax.set_xticklabels(new_tick_labels)

    callout = None              # player to highlight
    if callout: ax.scatter([rating_to_rank(callout.latest_rating())], [callout.get_ayd_rating()])

    ax.plot(whr_ranks,
             [lsq[0] * r + lsq[1] for r in whr_ranks],
             linewidth=0.2)
    # Maybe we should divide by std
    deltas = [(yd_ratings[i] - (lsq[0] * whr_ranks[i] + lsq[1])) for i in range(len(players))]
    abs_deltas = [abs(d) for d in deltas]
    delta_cutoff = sorted(abs_deltas, reverse=True)[9]
    abs_devs = [abs(deltas[i] / whr_stds[i]) for i in range(len(players))]
    dev_cutoff = sorted(abs_devs, reverse=True)[9]
    for (i, p) in enumerate(players):
        # if abs_deltas[i] >= delta_cutoff or p == callout:
        if abs_devs[i] >= dev_cutoff or p == callout:
            # print("{} is {}, should be {} to {} ".format(p.handle, yd_ratings[i], int(-deltas[i]),
            #                                             int(yd_ratings[i] - deltas[i])))
            ax.scatter([whr_ranks[i]], [yd_ratings[i]], s=4, c="orange")
            xytext = (-5,-5) if deltas[i] > 0 else (5,-5)
            horizontalalignment = "right" if deltas[i] > 0 else "left"
            ax.annotate(p.handle,
                         (whr_ranks[i], yd_ratings[i]),
                         xytext=xytext,
                         horizontalalignment=horizontalalignment,
                         textcoords="offset points")

    ax.set_xlabel("WHR rating")
    ax.set_ylabel("YD rating")
    plt.show()

print("Done.")
