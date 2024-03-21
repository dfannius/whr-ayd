from collections import namedtuple
from typing import Dict, List, Mapping, Optional, Set

# TODO
# + Predict result of game
# + Crosstables
# + Show rating changes for players with new games
# + Is incremental rating updating working?
# + Graph multiple players
# + Store players by ID rather than name
# + Case-insensitive handle lookup
# + List all games by a player
# + Always load data, no reason not to
# + Ranks on graph axes should be between ticks, not on them
#   + graph
#   + report
#   + whr-vs-yd
# + Refactor axis-labels-between-ticks code
# + Suppress wild swings while running algorithm so we can move the root games way back
# + In graphs, draw dotted lines or something during inactivity
# + Check historical prediction accuracy
# + After loading ratings, set new unpopulated ratings to a good default (most recent)
# + Refactor predict / predict_game
# + Give rank (not rating) credit for games played, somehow
#   + Smooth the_games_played_by
# + Remove obsolete command-line options
#   + Is --league / --leagues still useful? (No)
# - Convert everything to f-strings
# - Command-line option to compare two players
# - Why do I need RATING_FACTOR at all?
# - Don't draw ranks outside graph
# - Make handle lookup faster? Not really necessary right now
# - Do visual clusters in graph by season rather than date
# - Graph one or more players vs iterations as algorithm runs
# - Graph all players fitting some criterion (e.g., group)
# - Anchor players? (e.g., RMeigs)
#   This doesn't seem necessary now that I have GAMES_BONUS
# - Get historical YD ratings again
#   (The problem is that you can only get historical ratings through the player page)
# - Smarter choice of what player to update

ROOT_PLAYER_ID = -1

import argparse
import glob
import itertools
import math
import os
import random
import re
import sqlite3
from bs4 import BeautifulSoup, NavigableString
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from scipy import integrate
import seaborn as sns

parser = argparse.ArgumentParser(description="WHR for AYD")
parser.add_argument("--report-file", type=str, default="report.txt", metavar="F",
                    help="File with ratings report")
parser.add_argument("--parse-seasons", action="store_true", default=False,
                    help="Parse HTML season files into game data file")
parser.add_argument("--parse-season-start", type=int, default=1, metavar="N",
                    help="Season to start parsing HTML files from")
parser.add_argument("--parse-cycle-start", type=int, default=1, metavar="N",
                    help="Cycle (1-3) to start parsing HTML files from")
parser.add_argument("--analyze-games", action="store_true", default=False,
                    help="Analyze game data file")
parser.add_argument("--print-report", action="store_true", default=False,
                    help="Produce ratings report")
parser.add_argument("--draw-graph", type=str, default=None, metavar="H",
                    help="Handle of user's graph to draw")
parser.add_argument("--draw-graphs", type=str, nargs="+", default=None, metavar="H+",
                    help="Handle of multiple users' graph to draw")
parser.add_argument("--graph-names", action="store_true", default=False,
                    help="Show opponent names on graph")
parser.add_argument("--whr-vs-yd", action="store_true", default=False,
                    help="Draw scatterplot of WHR vs YD ratings")
parser.add_argument("--min-date", type=int, default=0, metavar="N",
                    help="Mininum active date for players in graph")
parser.add_argument("--note-new-games", action="store_true", default=False,
                    help="Print the results of newly parsed games")
parser.add_argument("--predict", type=str, nargs=2,
                    help="Supply the probability of one player beating another")
parser.add_argument("--report", action="store_true", default=False,
                    help="Draw graphical report of players' ratings")
parser.add_argument("--changes", action="store_true", default=False,
                    help="Produce report on ratings changes")
parser.add_argument("--xtable", action="store_true", default=False,
                    help="Produce crosstable report")
parser.add_argument("--list-games", type=str, default=None, metavar="H",
                    help="List the games of a player")
parser.add_argument("--prob-report", action="store_true", default=False,
                    help="Report on games' win probabilities")
parser.add_argument("--count-games-by", action="store_true", default=False,
                    help="Plot how many games have been played by current players over time")

args = parser.parse_args()

# I need to multiply rating differences by this much for predictions
# to be most accurate
RATING_FACTOR = 1.3

# If True, then after WHR has been performed, we calculate a 'average
# league strength' and use that to inflate everyone's user-friendly
# ranks accordingly.
#
# Average league strength at any given time is the average of the
# bonus points earned by all active players (at that time) up to that
# time.  One bonus point is earned per game played. Once we compute
# league strength (and it is smoothed out to avoid artifacts) it is
# applied to ranks at the rate of GAMES_BONUS.
#
# Of course this league strength is completely fictional, but:
#  - it acknowledges that players improve over time, especially in the YD environment
#  - it corresponds roughly to YD's own bonus point system (one YD rating point is
#    given per game played)
#  - it creates more reasonable looking results!
#
# A GAMES_BONUS of 0.04 means that if everyone in the league stuck
# around and played 25 more games (5 months, or 6.7 in calendar time
# including breaks), they'd all gain one rank on average. That seems
# pretty liberal, but it resulted in the best-looking graphs in
# practice.

BONUS_RANK = True

if BONUS_RANK:
    RATING_SCALE = 1.4
    RATING_SHIFT = -4.2
    GAMES_BONUS = 0.04
else:
    # For converting to AGA-ish ratings
    RATING_SCALE = 1.4
    RATING_SHIFT = -0.2
    GAMES_BONUS = 0

the_games_played_by = {}

def smooth_dict(dict_in: Dict[int, float]) -> Dict[int, float]:
    # Smooth a collection of timed values with a Gaussian filter.
    # Accounts for gaps in data.
    filter = { -3: 0.070,
               -2: 0.131,
               -1: 0.191,
               0: 0.216,
               1: 0.191,
               2: 0.131,
               3: 0.070 }
    dict_out = {}
    for k in dict_in.keys():
        num = 0
        denom = 0
        for dt in filter.keys():
            t = k + dt
            if t in dict_in:
                num += dict_in[t] * filter[dt]
                denom += filter[dt]
        dict_out[k] = num / denom
    return dict_out

def rating_to_rank(raw_r, d):
    """Convert a raw (internal) rating to something more AGA-like."""
    # To convert to AGA ratings it seems that we should divide raw_r
    # by 1.6, but to get ratings to match up at all at both the top
    # and bottom of the population I need to multiply instead.
    bonus = the_games_played_by[d] * GAMES_BONUS if d in the_games_played_by else 0
    return raw_r * RATING_SCALE + RATING_SHIFT + bonus

def rank_to_rank_str(rank, integral=False):
    """Return the string representation of a rank. If `integral` is true and the
    rank is an integer, don't add any decimals."""
    if integral and int(rank) == rank:
        rank = int(rank)
        if rank == 1:
            return "1k/1d"
        dan_fmt = "{}d"
        kyu_fmt = "{}k"
    else:
        dan_fmt = "{:.2f}d"
        kyu_fmt = "{:.2f}k"
    if rank >= 1:
        return dan_fmt.format(rank)
    else:
        return kyu_fmt.format(2-rank)

def rank_to_rank_str_tick(rank):
    """Return the appropriate string to label a range between ticks."""
    r = rank
    if r > 0: r += 1
    r = int(r)
    if r >= 1:
        ans = f"{r}d"
    else:
        ans = f"{1-r}k"
    # print(f"{rank} -> {ans}")
    return ans

def rating_to_rank_str(raw_r, d):
    """Return the string representation of a raw rating."""
    return rank_to_rank_str(rating_to_rank(raw_r, d))

def r_to_gamma(r):
    """Convert r (Elo-like rating) to gamma (Bradley-Terry-like rating)."""
    r = min(r, 20)           # handle some strange overflow
    return math.exp(r)

# A 'season' consists of three 'cycles' (each lasting one month) and
# then an off month. 'dates' are in units of months. So the date goes
# up by 4 with each new season.

def date_to_season_cycle(d):
    season = int(d/4)
    cycle = d - season * 4
    return (season, cycle)

def date_to_str(d):
    season, cycle = date_to_season_cycle(d)
    return "{}{}".format(season, "ABC"[cycle])

def season_cycle_to_date(s, c):
    return s * 4 + c

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

    def num_games(self):
        return len(self.wins) + len(self.losses)

    def __repr__(self):
        return "{}: {}".format(self.date, rating_to_rank_str(self.rating, self.date))

class Result:
    def __init__(self, date: int, handle: str, rating: float, won: bool):
        self.date: int = date        # date of game
        self.handle: str = handle    # handle of opponent
        self.rating: float = rating    # rating of opponent at that time
        self.rank: float = rating_to_rank(rating, date)
        self.sep_rank: float = self.rank # rank that may have been moved a bit to avoid overlap
        self.won: bool = won          # whether we beat them

class Player:
    def __init__(self, player_id, handle, player_db, is_root: bool = False):
        self.player_id: str = player_id
        self.handle: str = handle
        self.games: List[Game] = []
        self.yd_ratings: Dict[int, int] = {} # key is date
        self.rating_history: List[RatingDatum] = []
        self.player_db: PlayerDB = player_db
        self.root: bool = is_root
        self.rating_hash: Mapping[int, RatingDatum] = {}

    def __repr__(self):
        return self.handle

    def set_yd_rating(self, date: int, yd_rating: int):
        self.yd_ratings[date] = yd_rating

    def get_yd_rating(self, date: int) -> int:
        return self.yd_ratings[date]

    def get_latest_yd_rating(self) -> int:
        if self.yd_ratings:
            return self.yd_ratings[max(self.yd_ratings.keys())]
        else:
            return 0

    def include_in_graph(self, require_both_results=True) -> bool:
        if self.root: return False
        if len(self.rating_history) <= 1: return False
        if require_both_results and (len(self.get_wins()) == 0 or len(self.get_losses()) == 0):
            return False
        if self.handle == "Chimin": return False
#       if self.handle == "RMeigs": return False # throws YD calculations off too much
        return True

    def write_rating_history(self, f):
        print('"{}"'.format(self.handle), file=f, end="")
        print(',{}'.format(self.get_latest_yd_rating()), file=f, end="")
        print(',{}'.format(len(self.rating_history)), file=f, end="")
        for r in self.rating_history:
            print(',{},{:.8f},{:.8f}'.format(r.date, r.rating, r.std), file=f, end="")
        print(file=f)

    def add_rating(self, date: int, rating: float, std: float):
        for r in self.rating_history:
            if r.date == date:
                r.rating = rating
                r.gamma = r_to_gamma(rating)
                r.std = std
                return
        self.rating_history.append(RatingDatum(date, rating, std))

    def warm_start_new_ratings(self):
        # Update unpopulated new ratings to the latest one instead of 0
        latest_real_rating = 0
        for r in self.rating_history:
            if r.std == 0:
                r.rating = latest_real_rating
                self.rating_hash[r.date] = r
            else:
                latest_real_rating = r.rating

    def hash_ratings(self):
        for r in self.rating_history:
            self.rating_hash[r.date] = r

    def copy_rating_history_from(self, other: "Player"):
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

    def count_games_played(self, played_on: Dict[int, Set["Player"]]):
        self.games_up_to = {}
        d = -100
        games_this_date = 0
        total_games = 0
        for g in self.games:
            if g.winner.root or g.loser.root: continue
            if g.date not in played_on:
                played_on[g.date] = set()
            played_on[g.date].add(self)
            if g.date == d:
                games_this_date += 1
            else:
                if games_this_date > 0:
                    total_games += games_this_date
                    self.games_up_to[d] = total_games
                games_this_date = 1
                d = g.date
        if games_this_date > 0:
            total_games += games_this_date
            self.games_up_to[d] = total_games
            games_this_date = 0

    def latest_rating(self) -> float:
        # There may be elements of the rating history that don't have any games yet,
        # so we have to skip over them.
        for r in reversed(self.rating_history):
            if r.num_games() > 0 and r.std != 0:
                return r.rating
        return 0

    def latest_date(self) -> int:
        if self.root:
            return 0
        return self.rating_history[-1].date

    def latest_std(self) -> float:
        for r in reversed(self.rating_history):
            if r.num_games() > 0 and r.std != 0:
                return r.std
        return 10                # If no ratings, we don't know anything

    def get_std(self, date: int) -> float:
        for r in reversed(self.rating_history):
            if r.date == date and r.std != 0:
                return r.std
        return 10                # If no ratings, we don't know anything

    def get_rating_fast(self, date: int) -> float:
        if self.root:
            return 0
        else:
            return self.rating_hash[date].rating

    def get_gamma_fast(self, date: int) -> float:
        if self.root:
            return 1
        else:
            return self.rating_hash[date].gamma

    def get_results(self) -> List[Result]:
        results = []
        for r in self.rating_history:
            for w in r.wins:
                if not w.root:
                    results.append(Result(r.date, w.handle, w.get_rating(r.date), True))
            for l in r.losses:
                if not l.root:
                    results.append(Result(r.date, l.handle, l.get_rating(r.date), False))
        return results

    def get_wins(self) -> List[Result]:
        results = []
        for r in self.rating_history:
            for w in r.wins:
                if not w.root:
                    results.append(Result(r.date, w.handle, w.get_rating(r.date), True))
        return results

    def get_losses(self) -> List[Result]:
        results = []
        for r in self.rating_history:
            for l in r.losses:
                if not l.root:
                    results.append(Result(r.date, l.handle, l.get_rating(r.date), False))
        return results

    def get_wl_vs(self, other_player):
        (wins, losses) = (0, 0)
        for r in self.rating_history:
            wins += r.wins.count(other_player)
            losses += r.losses.count(other_player)
        return (wins, losses)

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
        # Add a virtual win and loss in the ancient past against a seed player
        # to avoid degenerate situations
        root_date = min_date - 2000
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
                assert self.rating_history[i].date == g.date
            if g.winner == self:
                self.rating_history[i].wins.append(g.loser)
            else:
                self.rating_history[i].losses.append(g.winner)

    # Return (Hessian, gradient)
    def compute_derivatives(self):
        # The WHR paper expresses w^2 in units of Elo^2/day. The conversion to r^2/month
        # means multiplying by (ln(10) / 400)^2 * 30 ~= 0.001
        # elo_wsq = 100         # I've also tried 300 but this looks good
        elo_wsq = 20            # but I think this may be even better!
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
                # Almost all elements of g and diagonal elements of H get
                # updated twice, once due to the relationship with the previous rating
                # and once due to the relationship with the following rating.
                g[i] -= dr * sigmasq_recip                   # Wiener
                H[i,i] -= sigmasq_recip                      # Wiener
                g[i-1] += dr * sigmasq_recip                 # Wiener
                H[i-1,i-1] -= sigmasq_recip                  # Wiener
                H[i-1,i] += sigmasq_recip                    # Wiener
                H[i,i-1] += sigmasq_recip                    # Wiener
        return (H, g)

    # Return magnitude of changes
    def iterate_whr(self, n: int) -> float:
        # print(f"iterate_whr {self.handle}")
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
            if abs(x[i]) >= 1:
                print(f"{n}: {self.handle} @ {r.date} changed by {-x[i]:.3f} from {r.rating:.3f}")
            if x[i] > 1.0: x[i] = 1.0
            elif x[i] < -1.0: x[i] = -1.0
            r.rating -= x[i]
            r.gamma = r_to_gamma(r.rating)

        return float(np.linalg.norm(x))

    # Return list of std deviation at each rating point
    def compute_stds(self):
        (H, _g) = self.compute_derivatives()
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
        return "{:03}: {} > {}".format(self.date, self.winner, self.loser)

class PlayerDB:
    def __init__(self) -> None:
        self.player_map: Dict[int, Player] = {}
        self.root_player: Player = self.get_player(ROOT_PLAYER_ID, "[root]", is_root=True)

    def get_player(self, player_id: int, handle: Optional[str]=None, is_root=False) -> Player:
        if player_id in self.player_map:
            player = self.player_map[player_id]
            if handle is not None:
                if player.handle is not None and player.handle != handle:
                    print(f"Player {player_id} was {player.handle}, now {handle}")
                    player.handle = handle
            return player
        else:
            player = Player(player_id, handle, self, is_root)
            self.player_map[player_id] = player
            return player

    def get_player_by_handle(self, handle:str) -> Player:
        h = handle.casefold()
        for p in self.player_map.values():
            if p.handle.casefold() == h:
                return p
        raise Exception(f"Couldn't find {handle}")

    def get_root_player(self) -> Player:
        return self.root_player

    def copy_rating_history_from(self, other_db: "PlayerDB"):
        for (player_id, player) in self.player_map.items():
            if player_id in other_db.player_map:
                player.copy_rating_history_from(other_db.player_map[player_id])

    def remove_recent_games(self, start_date: int):
        for p in self.player_map.values():
            p.remove_recent_games(start_date)

    def __getitem__(self, player_id: int) -> Player:
        return self.player_map[player_id]

    def __len__(self) -> int:
        return len(self.player_map)

    def clear(self):
        self.player_map.clear()

    def values(self) -> List[Player]:
        return list(self.player_map.values())

    def count_games_played(self):
        played_on = {}
        played_by = {}
        for p in self.player_map.values():
            p.count_games_played(played_on)
        for d in played_on.keys():
            num_games_played_by = 0
            for p in played_on[d]:
                num_games_played_by += p.games_up_to[d]
            played_by[d] = num_games_played_by / len(played_on[d])
        return played_by

the_player_db = PlayerDB()

def is_cycle_name(tag):
    return (tag.name == "b" and tag.contents[0].startswith("AYD")
            or
            tag.name == "h3" and "League" in tag.contents[0])

# A full cycle name is something like "AYD League B, March 2014".
# Get just the date part
def cycle_to_date(s: str) -> str:
    if "," in s:
        return s.split(", ")[-1]
    else:
        return " ".join(s.split(" ")[-2:])

def flush_old_games(player_db: PlayerDB, start_season: int, start_cycle: int, old_games: List[Game]):
    start_date = season_cycle_to_date(start_season, start_cycle-1)
    player_db.remove_recent_games(start_date)
    games = [g for g in old_games if g.date < start_date]
    flushed_games = [g for g in old_games if g.date >= start_date]
    return games, flushed_games

def get_crosstable_tag(starting_at_tag):
    tag = starting_at_tag.previous_sibling
    while tag is not None:
        if type(tag) is NavigableString or tag.name != "h3":
            tag = tag.previous_sibling
        else:
            return tag
    return None

def parse_seasons(player_db: PlayerDB,
                  league: str,
                  start_season: int,
                  start_cycle: int,
                  existing_games: List[Game] ,
                  flushed_games: List[Game]):
    start_date = season_cycle_to_date(start_season, start_cycle-1)

    # An overview file may contain all three cycles of a season (AYD,
    # early EYD seasons) or a single cycle (late EYD seasons).
    overview_files = glob.glob("{}-overviews/*-overview.html".format(league))
    overview_file_array: List[List[str]] = []
    overview_file_re = re.compile(r"(\d+)-([^-]*)-overview.html")
    for fn in overview_files:
        match = re.search(overview_file_re, fn)
        if match:
            date = int(match.group(1))
            while date >= len(overview_file_array):
                overview_file_array.append([])
            overview_file_array[date].append(fn)

    # player_db.remove_recent_games(start_date)
    # games = [g for g in existing_games if g.date < start_date]
    games = existing_games
    new_games: List[Game] = []

    anchor_date = start_date
    while anchor_date < len(overview_file_array):
        print("{}...".format(anchor_date), end="", flush=True)
        for fn in overview_file_array[anchor_date]:
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
                    # print("CROSSTABLE")
                    # print(crosstable)
                    parent = crosstable.find_parent("table")
                    # print("PARENT")
                    # print(parent)
                    crosstable_name = get_crosstable_tag(parent)
                    if crosstable_name is None:
                        crosstable_name = get_crosstable_tag(parent.find_parent("table"))
                    # print(f"crosstable_name {crosstable_name}")
                    crosstable_date = cycle_to_date(crosstable_name.contents[0])
                    date = anchor_date + season_cycles.index(crosstable_date)
                    # print(f"crosstable_date = {crosstable_date}, date = {date}")

                    # Construct the list of players, in order
                    crosstable_players = []
                    trs = crosstable.find_all("tr")
                    for tr in trs:
                        tds = tr.find_all("td")
                        if len(tds) > 0:
                            #XX print(tds)
                            #XX name = tds[1].nobr.a.contents[0]
                            #XX handle = tds[2].contents[0]
                            handle = tds[1].a.contents[0]
                            m = re.search(r'id=(\d+)', tds[1].a.get('href'))
                            if m:
                                player_id = int(m.group(1))
                            else:
                                raise Exception(f"Couldn't find player_id for {handle}")
                            #XX print(handle)
                            #XX yd_rating = int(tds[11].contents[0])
                            yd_rating = int(tds[-1].contents[0])
                            player = player_db.get_player(player_id, handle)
                            player.set_yd_rating(date, yd_rating)
                            crosstable_players.append(player)

                    # Parse game results
                    row_player_idx = 0
                    for tr in trs:
                        tds = tr.find_all("td")
                        if len(tds) > 0:
                            for col_player_idx in range(row_player_idx + 1, len(crosstable_players)):
                                gif = tds[2+col_player_idx].find("img")["src"]
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

    return new_games

def init_whr(player_db: PlayerDB):
    for p in player_db.values():
        p.init_rating_history()

def iterate_whr(player_db: PlayerDB, n: int):
    max_change = -1.0
    players = list(player_db.values())
    random.shuffle(players)
    for p in players:
        change = abs(p.iterate_whr(n))
        if change > max_change:
            max_change = change
    return max_change

def run_whr(player_db: PlayerDB):
    for i in range(1000):
        if (i+1) % 100 == 0:
            print("{}...".format(i+1), end="", flush=True)
            # print("ITERATION {}".format(i))
        max_change = iterate_whr(player_db, i)
        if max_change < 1e-05:
            print("Completed WHR in {} iteration{}...".format(i+1, "s" if i > 0 else ""),
                  end="", flush=True)
            break
    for p in player_db.values():
        p.compute_stds()

def predict_prob(r1: float, s1: float, r2: float, s2: float):
    # Inputs: rating and std. dev. of each player
    mu_diff = (r1 - r2) * RATING_FACTOR
    var = s1 ** 2 + s2 ** 2
    if var == 0:
        print(f"No variance!")
        var = 2
    return integrate.quad(lambda d: 1. / (1 + np.exp(-d)) * np.exp(-(d - mu_diff)**2 / (2 * var)), -100, 100)[0] * (1. / math.sqrt(2 * math.pi * var))

def predict(p1: Player, p2: Player, date: int):
    r1 = p1.latest_rating()
    s1 = p1.latest_std()
    r2 = p2.latest_rating()
    s2 = p2.latest_std()
    d = max(p1.latest_date(), p2.latest_date())
    prob = predict_prob(r1, s1, r2, s2)
    p1_rank_str = rating_to_rank_str(r1, d)
    p1_std = s1 * RATING_SCALE
    p2_rank_str = rating_to_rank_str(r2, d)
    p2_std = s2 * RATING_SCALE
    return (p1_rank_str, p1_std, p2_rank_str, p2_std, prob)

def predict_game(g: Game):
    d = g.date
    w = g.winner
    l = g.loser
    return predict_prob(w.get_rating_fast(d), w.get_std(d), l.get_rating_fast(d), l.get_std(d))

def print_report(player_db: PlayerDB, fname: str):
    with open(fname, "w", encoding="utf-8") as f:
        players = [ p for p in player_db.values() if (not p.root) and p.latest_date() >= args.min_date ]
        for p in sorted(players,
                        key=lambda p: rating_to_rank(p.latest_rating(), p.latest_date()),
                        reverse=True):
            if len(p.rating_history) > 0:
                print("{:<10} {:>6} ± {:.2f}: {}".format(p.handle,
                                                         rating_to_rank_str(p.latest_rating(), p.latest_date()),
                                                         p.latest_std() * RATING_SCALE,
                                                         p.rating_history[1:]),
                      file=f)

def save_rating_history(player_db: PlayerDB, fname: str):
    with open(fname, "w", encoding="utf-8") as f:
        for p in sorted(player_db.values(), key=lambda p: p.latest_rating(), reverse=True):
            p.write_rating_history(f)

DB_NAME = "whr.db"

db_con = sqlite3.connect(DB_NAME)

def store_games(games: List[Game]):
    cur = db_con.cursor()
    data = [ (g.date, g.winner.player_id, g.loser.player_id) for g in games ]
    cur.executemany("INSERT OR REPLACE INTO games VALUES(?, ?, ?)", data)
    db_con.commit()
    cur.close()

def load_games(player_db: PlayerDB) -> List[Game]:
    games = []
    cur = db_con.cursor()
    cur.execute("SELECT * from games")
    rows = cur.fetchall()
    for (date, winner_id, loser_id) in rows:
        winner = player_db.get_player(winner_id)
        loser = player_db.get_player(loser_id)
        # print(f"{date} {winner_id} {winner.handle} > {loser_id} {loser.handle}")
        game = Game(date, winner, loser)
        games.append(game)
        winner.add_game(game)
        loser.add_game(game)
    db_con.commit()
    cur.close()
    return games

def store_ratings(player_db: PlayerDB):
    cur = db_con.cursor()
    data = [ (p.player_id, r.date, r.rating, r.std) for p in player_db.values() for r in p.rating_history ]
    cur.executemany("INSERT OR REPLACE INTO ratings VALUES(?, ?, ?, ?)", data)
    db_con.commit()
    cur.close()

def load_ratings(player_db: PlayerDB):
    old_player_db = PlayerDB()
    cur = db_con.cursor()
    cur.execute("SELECT * from ratings")
    rows = cur.fetchall()
    for (player_id, date, rating, std) in rows:
        p = old_player_db.get_player(player_id)
        p.add_rating(date, rating, std)
    for player in old_player_db.values():
        player.hash_ratings()
    player_db.copy_rating_history_from(old_player_db)
    for player in player_db.values():
        player.warm_start_new_ratings()
    db_con.commit()
    cur.close()

def store_yd_ratings(player_db: PlayerDB):
    cur = db_con.cursor()
    data = [ (p.player_id, date, yd_rating) for p in player_db.values() for (date, yd_rating) in p.yd_ratings.items() ]
    cur.executemany("INSERT OR REPLACE INTO yd_ratings VALUES(?, ?, ?)", data)
    db_con.commit()
    cur.close()

def load_yd_ratings(player_db: PlayerDB):
    cur = db_con.cursor()
    cur.execute("SELECT * from yd_ratings")
    rows = cur.fetchall()
    for (player_id, date, yd_rating) in rows:
        p = player_db.get_player(player_id)
        p.set_yd_rating(date, yd_rating)
    db_con.commit()
    cur.close()

def store_ids(player_db: PlayerDB):
    cur = db_con.cursor()
    data = [ (p.player_id, p.handle) for p in player_db.values() ]
    cur.executemany("INSERT OR REPLACE INTO ids VALUES(?, ?)", data)
    db_con.commit()
    cur.close()

def load_ids(player_db: PlayerDB):
    cur = db_con.cursor()
    cur.execute("SELECT * from ids")
    rows = cur.fetchall()
    for (player_id, handle) in rows:
        p = player_db.get_player(player_id, handle)
        # print(f"{player_id} {handle}")
    db_con.commit()
    cur.close()

NewGameStats = namedtuple("NewGameStats", ["p1", "p1_rank_str", "p1_std", "p2", "p2_rank_str", "p2_std", "prob"])

def compute_new_game_stats(new_games: List[Game]):
    new_game_stats = []
    for g in new_games:
        (p1_rank_str, p1_std, p2_rank_str, p2_std, prob) = predict(g.winner, g.loser, g.date)
        new_game_stats.append(NewGameStats(p1=g.winner, p1_rank_str=p1_rank_str, p1_std=p1_std,
                                           p2=g.loser, p2_rank_str=p2_rank_str, p2_std=p2_std,
                                           prob=prob))
        # print(f"   {g.winner.handle} ({p1_rank_str} ± {p1_std:.2f}) > {g.loser.handle} ({p2_rank_str} ± {p2_std:.2f}): ({prob*100:.3}% chance)")
    return new_game_stats

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

def sort_results(results: List[Result]) -> List[Result]:
    return sorted(results, key = lambda r: (r.date, r.rating))

def separate_clusters(clusters, delta):
    ok = False
    while not ok:
        ok = True
        last_val = None
        last_date = None
        i = 0
        while i < len(clusters):
            if last_val is not None:
                if clusters[i][0][2] == last_date and clusters[i][0][1] < last_val + delta:
                    clusters[i-1].extend(clusters[i])
                    clusters = clusters[0:i] + clusters[i+1:]
                    clust = clusters[i-1]   # cluster to update
                    mean = sum(p[0] for p in clust) / len(clust)
                    lowest = mean - delta * (len(clust) - 1) / 2
                    for (j, p) in enumerate(clust):
                        p[1] = lowest + j * delta
                    last_val = clusters[i-1][-1][1]
                    continue    # don't increment i
            last_val = clusters[i][-1][1]
            last_date = clusters[i][-1][2]
            i += 1

    return clusters

def separate(results: List[Result], delta):
    clusters = [[[results[i].rank, results[i].rank, results[i].date]]
                for i in range(len(results))] # (orig_val, new_val, date)
    clusters = separate_clusters(clusters, delta)
    pairs = list(itertools.chain.from_iterable(clusters))
    vals = [pair[1] for pair in pairs]
    for (i, r) in enumerate(results):
        r.sep_rank = vals[i]

def draw_opponents(games, color, draw_names, plotted_ranks):
    if len(games) > 0:
        dates = [w.date for w in games]
        handles = [w.handle for w in games]
        ratings = [w.rating for w in games]
        sep_ranks = [w.sep_rank for w in games]
        ranks = [rating_to_rank(r, d) for (r, d) in zip(ratings, dates)]
        plotted_ranks += ranks
        plt.scatter(dates, ranks, edgecolors=color, facecolors="none", marker="o")
        if draw_names:
            for (i, handle) in enumerate(handles):
                plt.annotate(handle,
                             xy=(dates[i], sep_ranks[i]),
                             xytext=(5, 0),
                             textcoords="offset points",
                             fontsize="x-small", verticalalignment="center", color=color)

def label_ranks_x(tick_vals):
    plt.gca().tick_params(axis='x', which='both', labelbottom=False)
    new_tick_vals = [x - 0.5 for x in tick_vals][1:]
    new_tick_labels = [rank_to_rank_str_tick(r-1) for r in new_tick_vals]
    ylim = plt.gca().get_ylim()
    for position, label in zip(new_tick_vals, new_tick_labels):
        plt.text(position,
                 ylim[0] - 0.05 * (ylim[1] - ylim[0]),
                 label, rotation=0, ha='center', va='top')

def label_ranks_y(tick_vals):
    plt.gca().tick_params(axis='y', which='both', left=False, right=False, labelleft=False)
    new_tick_vals = [y - 0.5 for y in tick_vals][1:]
    new_tick_labels = [rank_to_rank_str_tick(r-1) for r in new_tick_vals]
    xlim = plt.gca().get_xlim()
    for position, label in zip(new_tick_vals, new_tick_labels):
        plt.text(xlim[0] - 0.05 * (xlim[1] - xlim[0]),
                 position,
                 label, rotation=0, ha='center', va='top')

def plot_rank_line(dates, ranks, handle):
    groups = []
    for (d, r) in zip(dates, ranks):
        if len(groups) == 0:
            groups = [[(d, r)]]
            continue
        delta = d - groups[-1][-1][0]
        if delta > 2 or (delta == 2 and d % 4 != 0):
            groups.append([(d, r)])
        else:
            groups[-1].append((d,r))
    line, = plt.plot([d for (d,r) in groups[0]], [r for (d,r) in groups[0]])
    color = line.get_color()
    for n in range(1, len(groups)):
        plt.plot([groups[n-1][-1][0], groups[n][0][0]],
                 [groups[n-1][-1][1], groups[n][0][1]],
                 color=color, linestyle=':')
        if n == len(groups) - 1:
            plt.plot([d for (d,r) in groups[n]],
                     [r for (d,r) in groups[n]],
                     color=color,
                     label=handle)
        else:
            plt.plot([d for (d,r) in groups[n]],
                     [r for (d,r) in groups[n]],
                     color=color)
    return color

def do_draw_graphs():
    plot_dir = "plots"
    if not os.path.exists(plot_dir):
        os.mkdir(plot_dir)

    handles = [args.draw_graph] if args.draw_graph else args.draw_graphs
    multiple_handles = len(handles) > 1
    draw_names = (not multiple_handles) and args.graph_names
    handle_colors = {}
    last_dates = {}
    last_ranks = {}

    plt.figure(figsize=(8,12))
    all_dates = []
    all_plotted_ranks = []

    if args.draw_graph:
        plt.title("\n" + args.draw_graph + "\n")

    for handle in handles:
        p = the_player_db.get_player_by_handle(handle)
        history = p.rating_history[1:]
        dates = [r.date for r in history]
        all_dates = list(set(all_dates + dates))
        ratings = [r.rating for r in history]

        ranks = [rating_to_rank(r.rating, r.date) for r in history]
        plotted_ranks = ranks
        if len(dates) > 1:
            if args.draw_graph:
                plot_low_ranks = [rating_to_rank(r.rating - r.std, r.date) for r in history]
                plot_high_ranks = [rating_to_rank(r.rating + r.std, r.date) for r in history]
                plt.fill_between(dates,
                                 plot_low_ranks,
                                 plot_high_ranks,
                                 alpha=0.2)
            handle_colors[handle] = plot_rank_line(dates, ranks, handle)
        elif len(dates) == 1:
            # Special hacky code for when we have only one date, so we don't
            # draw an invisibly thin rectangle.
            plt.xlim(dates[0] - 1, dates[0] + 1)
            radius = 0.01
            plot_dates = [dates[0] - radius, dates[0] + radius]
            plot_ranks = [ranks[0], ranks[0]]
            if args.draw_graph:
                plot_low_rank = rating_to_rank(history[0].rating - history[0].std, history[0].date)
                plot_high_rank = rating_to_rank(history[0].rating + history[0].std, history[0].date)
                plot_low_ranks = [plot_low_rank] * 2
                plot_high_ranks = [plot_high_rank] * 2
                plt.fill_between(plot_dates,
                                 plot_low_ranks,
                                 plot_high_ranks,
                                 alpha=0.2)
            line, = plt.plot(plot_dates, plot_ranks, label=handle)
            handle_colors[handle] = line.get_color()

        if dates:
            last_dates[handle] = dates[-1]
            last_ranks[handle] = ranks[-1]

        if args.draw_graph:
            results = sort_results(p.get_results())
            max_rating = max(r.rating for r in results)
            min_rating = min(r.rating for r in results)
            rating_spread = max_rating - min_rating
            separate(results, rating_spread / 50)
            wins = [r for r in results if r.won]
            losses = [r for r in results if not r.won]
            draw_opponents(wins, "green", draw_names, plotted_ranks)
            draw_opponents(losses, "red", draw_names, plotted_ranks)

        all_plotted_ranks += plotted_ranks

    if args.draw_graphs:
        cs = [[last_ranks[h], last_ranks[h], last_dates[h], h] for h in handles]
        cs.sort(key=lambda x: (x[2], x[0]))
        clusters = [[c] for c in cs] # (orig_val, new_val, date)
        clusters = separate_clusters(clusters, 0.05)
        vals = list(itertools.chain.from_iterable(clusters))
        for v in vals:
            plt.annotate(v[3],
                         xy=(v[2], v[1]),
                         xytext=(5,0),
                         textcoords="offset points",
                         fontsize="x-small",
                         verticalalignment="center",
                         color=handle_colors[v[3]])


    y_min = int(min(all_plotted_ranks) - 1)
    y_max = int(max(all_plotted_ranks) + 1)
    plt.ylim(y_min, y_max)

    new_tick_vals = np.arange(y_min, y_max + 1, 1.0)
    new_tick_labels = [rank_to_rank_str(r, True) for r in new_tick_vals]
    plt.xticks(all_dates, date_str_ticks(all_dates))
    plt.yticks(new_tick_vals)
    label_ranks_y(new_tick_vals)

    plt.xlabel("Season")
    plt.ylabel("Rank")
    plt.savefig("{}/{}.png".format(plot_dir, handle))
    plt.tight_layout()
    plt.show()

def do_list_games():
    p = the_player_db.get_player_by_handle(args.list_games)
    print(f"Games of {p.handle} ({p.player_id}):")
    for g in p.games:
        # if g.winner.root or g.loser.root: continue
        w_rating = rating_to_rank_str(g.winner.get_rating_fast(g.date), g.date)
        l_rating = rating_to_rank_str(g.loser.get_rating_fast(g.date), g.date)
        prob = predict_game(g)
        if g.winner == p:
            print(f"{g.date:3}: {w_rating} W   {g.loser.handle:10} ({l_rating})          {100*prob:.1f}%")
        else:
            print(f"{g.date:3}: {l_rating}   l {g.winner.handle:10}         ({w_rating})         {100-100*prob:.1f}%")

def do_prob_report():
    probs = []
    for p in the_player_db.values():
        for g in p.games:
            if g.winner == p and not g.loser.root:
                probs.append(predict_game(g))
    probs.sort()
    BINS = 20
    xs = []
    ys = []
    for n in range(BINS):
        min_prob = n / BINS
        max_prob = (n+1) / BINS
        rights = [ p for p in probs if min_prob <= p and p < max_prob ]
        wrongs = [ p for p in probs if min_prob <= 1-p and 1-p < max_prob ]
        if len(rights) + len(wrongs) == 0:
            sucess = 0.5
        else:
            success = len(rights) / (len(rights) + len (wrongs))
        xs.append((n+0.5) / BINS)
        ys.append(success)
    plt.figure(figsize=(8,8))
    plt.title("\nWin probability\n")
    plt.plot(xs, ys, 'o')
    plt.plot([0, 1], [0, 1], linewidth=0.5)
    plt.xlabel("Predicted")
    plt.ylabel("Observed")
    plt.show()

def do_count_games_by(games_played_by):
    dates = sorted(games_played_by.keys())
    plt.figure(figsize=(8,8))
    plt.title("\nAverage cumulative games played by active players\n")
    plt.plot(dates, [games_played_by[d] for d in dates], 'o')
    plt.show()

def do_whr_vs_yd():
    players = [p for p in the_player_db.values() if p.include_in_graph()]
    players = [p for p in players if p.rating_history[-1].date >= args.min_date]
    whr_ranks = [rating_to_rank(p.latest_rating(), p.latest_date()) for p in players]
    whr_stds = [RATING_SCALE * p.latest_std() for p in players]
    yd_ratings = [p.get_latest_yd_rating() for p in players]

    W = np.vstack([whr_ranks, np.ones(len(whr_ranks))]).T
    (lsq, _resid, _rank, _sing) = np.linalg.lstsq(W, yd_ratings, rcond=None)

    _fig, ax = plt.subplots()
    ax.scatter(whr_ranks, yd_ratings, s=4)
    ax.errorbar(whr_ranks, yd_ratings, xerr=whr_stds, fmt="none", linewidth=0.2)
    ax.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(base=1))
    ax.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(base=100))
    tick_vals = ax.get_xticks()
    label_ranks_x(tick_vals)

    callout = None
    if callout: ax.scatter([rating_to_rank(callout.latest_rating(), callout.latest_date())], [callout.get_latest_yd_rating()])

    # ax.plot(whr_ranks,
    #          [lsq[0] * r + lsq[1] for r in whr_ranks],
    #          linewidth=0.2)
    # Maybe we should divide by std
    deltas = [(yd_ratings[i] - (lsq[0] * whr_ranks[i] + lsq[1])) for i in range(len(players))]
    num_callouts = 40
    abs_devs = [abs(deltas[i] / whr_stds[i]) for i in range(len(players))]
    dev_cutoff = sorted(abs_devs, reverse=True)[num_callouts-1]
    for (i, p) in enumerate(players):
        if abs_devs[i] >= dev_cutoff or p == callout:
            # print("{} is {}, should be {} to {} ".format(p.handle, yd_ratings[i], int(-deltas[i]),
            #                                             int(yd_ratings[i] - deltas[i])))
            if p == callout:
                color = "yellow"
            elif deltas[i] < 0:
                color = "orange"
            else:
                color= "red"
            ax.scatter([whr_ranks[i]], [yd_ratings[i]], s=4, c=color)
            xytext = (-5,-5) if deltas[i] > 0 else (5,-5)
            horizontalalignment = "right" if deltas[i] > 0 else "left"
            ax.annotate(p.handle,
                         (whr_ranks[i], yd_ratings[i]),
                         xytext=xytext,
                         horizontalalignment=horizontalalignment,
                         textcoords="offset points",
                         fontsize="x-small")

    ax.set_xlabel("WHR rating")
    ax.set_ylabel("YD rating")
    plt.show()

def do_predict():
    p1_handle = args.predict[0]
    p2_handle = args.predict[1]
    p1 = the_player_db.get_player_by_handle(p1_handle)
    p2 = the_player_db.get_player_by_handle(p2_handle)
    (p1_rank_str, p1_std, p2_rank_str, p2_std, prob) = predict(p1, p2, max(p1.latest_date(), p2.latest_date()))
    print(f"The probability of {p1_handle} ({p1_rank_str} ± {p1_std:.2f}) beating {p2_handle} ({p2_rank_str} ± {p2_std:.2f}) is {prob*100:.3}%.")

def do_report():
    # TODO: combine with whr_vs_yd
    players = [p for p in the_player_db.values() if p.include_in_graph(True)]
    players = [p for p in players if p.rating_history[-1].date >= args.min_date]
    players.sort(key=lambda p: p.latest_rating())
    whr_ranks = [rating_to_rank(p.latest_rating(), p.latest_date()) for p in players]
    whr_stds = [RATING_SCALE * p.latest_std() for p in players]

    y_poses = range(1, len(players)+1)
    _fig, ax = plt.subplots(figsize=(8,12))

    ax.get_yaxis().set_ticks([])
    for (i, p) in enumerate(players):
        # ax.text(-0.5, y_poses[i], p.handle)
        ax.annotate(p.handle,
                    xy=(whr_ranks[i] - whr_stds[i] - 0.5, y_poses[i]),
                    fontsize="x-small",
                    verticalalignment="center",
                    horizontalalignment="right")
    ax.scatter(whr_ranks, y_poses, s=4)
    ax.errorbar(whr_ranks, y_poses, xerr=whr_stds, linewidth=0.2, fmt="none")
    for x in (-13, -8, -3, 1, 5,):
        ax.axvline(x, color="black", linewidth=0.2)

    ax.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(base=1))
    tick_vals = ax.get_xticks()
    label_ranks_x(tick_vals)

    plt.show()

def do_changes():
    players = [p for p in the_player_db.values() if p.include_in_graph(True)]
    slopes = []
    for p in players:
        earliest_date = p.rating_history[1].date
        latest_date = p.rating_history[-1].date
        earliest_rating = p.rating_history[1].rating
        latest_rating = p.rating_history[-1].rating
        date_delta = latest_date - earliest_date
        rating_delta = latest_rating - earliest_rating
        rating_slope = 0 if date_delta == 0 else rating_delta / date_delta
        slopes.append( (p, rating_slope) )
        # print(f"{p.handle}: {earliest_rating:.2f} on {earliest_date} to {latest_rating:.2f} on {latest_date}, slope {rating_slope:.3f}")
    slopes.sort(key=lambda x:x[1], reverse=True)
    for (p, slope) in slopes:
        print(f"{p.handle:10} {slope:.3f}")
    avg_slope = sum(slope for (p, slope) in slopes) / len(slopes)
    print(f"Average slope {avg_slope:.3f}")

def do_xtable():
    players = [p for p in the_player_db.values() if p.rating_history and p.rating_history[-1].date >= args.min_date]
    players.sort(key=lambda p: p.latest_rating(), reverse=True)
    whr_ranks = [rating_to_rank(p.latest_rating(), p.latest_date()) for p in players]
    print()
    print("                      ", end="")
    for (i, p1) in enumerate(players):
        inits = p1.handle
        print(f"{inits[:3]} ", end="")
        p1.inits = inits
    print()
    for (i, p1) in enumerate(players):
        print(f"{p1.handle:10s} {rank_to_rank_str(whr_ranks[i]):>6s} {p1.inits[:3]} ", end="")
        (wins_vs_stronger, losses_vs_stronger, wins_vs_weaker, losses_vs_weaker) = (0, 0, 0, 0)
        for (j, p2) in enumerate(players):
            if p1 == p2:
                print("--- ", end="")
                continue
            (wins, losses) = p1.get_wl_vs(p2)
            if wins + losses > 0:
                if j < i:
                    wins_vs_stronger += wins
                    losses_vs_stronger += losses
                else:
                    wins_vs_weaker += wins
                    losses_vs_weaker += losses
                if wins >= 10: wins = "^"
                if losses >= 10: losses = "^"
                print(f"{wins}-{losses} ", end="")
            else:
                print("    ", end="")
        print(f"  {wins_vs_stronger:2d}-{losses_vs_stronger:2d} {wins_vs_weaker:2d}-{losses_vs_weaker:2d}")

def run() -> None:
    # print("Loading ids...", end="", flush=True)
    load_ids(the_player_db)

    # print("Loading games...", end="", flush=True)
    games: List[Game] = load_games(the_player_db)
    new_games: List[Game] = []

    # print("Loading YD rating history...", end="", flush=True)
    load_yd_ratings(the_player_db)

    if args.parse_seasons:
        print("Parsing seasons...", end="", flush=True)
        games, flushed_games = flush_old_games(the_player_db,
                                               args.parse_season_start,
                                               args.parse_cycle_start,
                                               games)
        for league in ["ayd", "eyd"]:
            print("{}...".format(league), end="")
            these_new_games = parse_seasons(the_player_db,
                                            league,
                                            args.parse_season_start,
                                            args.parse_cycle_start,
                                            games,
                                            flushed_games)
            new_games.extend(these_new_games)
        store_games(games)

    # print("Storing ids...", end="", flush=True)
    store_ids(the_player_db)

    # print("Storing YD rating history...", end="", flush=True)
    store_yd_ratings(the_player_db)

    init_whr(the_player_db)

    # print("Loading rating history...", end="", flush=True)
    load_ratings(the_player_db)

    new_game_stats = []

    global the_games_played_by
    the_games_played_by = smooth_dict(the_player_db.count_games_played())
    # for d in sorted(the_games_played_by.keys()):
    #     print(f"{d}: {the_games_played_by[d]:.2f}")

    if new_games and args.note_new_games:
        new_game_stats = compute_new_game_stats(new_games)
        num_games_str = "1 new game" if len(new_games) == 1 else f"{len(new_games)} new games"
        print(f"Processing {num_games_str}...", end="", flush=True)

    if args.analyze_games:
        print("Running WHR...", end="", flush=True)
        run_whr(the_player_db)
        # print("Storing rating history...", end="", flush=True)
        store_ratings(the_player_db)
        print("Done.")

    db_con.close()

    if new_games and args.note_new_games:
        print("New games:")
        for gs in new_game_stats:
            p1_rank_str = rating_to_rank_str(gs.p1.latest_rating(), gs.p1.latest_date())
            p1_std = gs.p1.latest_std() * RATING_SCALE
            p2_rank_str = rating_to_rank_str(gs.p2.latest_rating(), gs.p2.latest_date())
            p2_std = gs.p2.latest_std() * RATING_SCALE
            print(f"   {gs.p1.handle:10} ({gs.p1_rank_str:>6} ± {gs.p1_std:.2f} -> {p1_rank_str:>6} ± {p1_std:.2f}) > ", end="")
            print(f"{gs.p2.handle:10} ({gs.p2_rank_str:>6} ± {gs.p2_std:.2f} -> {p2_rank_str:>6} ± {p2_std:.2f}) ({gs.prob*100:.3}% chance)")

    if args.print_report:
        print("Printing report...", end="", flush=True)
        print_report(the_player_db, args.report_file)
        print("Done.")

    sns.set_theme()

    if args.draw_graph or args.draw_graphs:
        do_draw_graphs()
    if args.list_games:
        do_list_games()
    if args.whr_vs_yd:
        do_whr_vs_yd()
    if args.predict:
        do_predict()
    if args.report:
        do_report()
    if args.changes:
        do_changes()
    if args.xtable:
        do_xtable()
    if args.prob_report:
        do_prob_report()
    if args.count_games_by:
        do_count_games_by(the_games_played_by)

run()
