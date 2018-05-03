import argparse
from bs4 import BeautifulSoup, NavigableString
import csv
import math
import numpy as np

parser = argparse.ArgumentParser(description='WHR for AYD')
parser.add_argument("--games-file", type=str, default="games.csv", metavar="F",
                    help="File of game data")
parser.add_argument("--ratings-file", type=str, default="ratings.csv", metavar="F",
                    help="File of ratings data")
parser.add_argument("--parse-seasons", action="store_true", default=False,
                    help="Parse HTML season files into game data file")
parser.add_argument("--analyze-games", action="store_true", default=False,
                    help="Analyze game data file")
parser.add_argument("--print-ratings", action="store_true", default=False,
                    help="Print ratings data")
parser.add_argument("--load-ratings", action="store_true", default=False,
                    help="Load data from ratings file")

args = parser.parse_args()

def rating_to_rank(raw_r):
    # To convert to AGA ratings it seems that we should divide raw_r
    # by 1.6, but that compresses ranks more than I like.
    r = raw_r - 1
    if r >= 1:
        return "{:.2f}d".format(r)
    else:
        return "{:.2f}k".format(2-r)

DEBUG = False
def dprint(*args, **kwargs):
    if DEBUG:
        print(*args, **kwargs)

def is_cycle_name(tag):
    return tag.name == "b" and tag.contents[0].startswith("AYD")

def r_to_gamma(r):
    return math.exp(r)

class RatingDatum:
    def __init__(self, date, rating, std=0):
        self.date = date
        self.rating = rating
        self.std = std
        self.wins = []
        self.losses = []

    def set_std(self, std):
        self.std = std

    def __repr__(self):
        return "{}: {}".format(self.date, rating_to_rank(self.rating))

class Player:
    def __init__(self, name, handle, is_root=False):
        self.name = name
        self.handle = handle
        self.games = []
        self.rating_history = []
        self.root = is_root

    def __repr__(self):
        return "{} ({})".format(self.name, self.handle)

    def write_rating_history(self, f):
        print('"{}","{}","{}"'.format(self.name, self.handle, len(self.rating_history)), file=f, end="")
        for r in self.rating_history:
            print(',"{}","{}","{}"'.format(r.date, r.rating, r.std), file=f, end="")
        print(file=f)

    def read_rating_history(self, row):
        num_points = int(row[0])
        for i in range(num_points):
            self.rating_history.append(RatingDatum(int(row[3*i+1]), float(row[3*i+2]), float(row[3*i+3])))

    def add_game(self, game):
        self.games.append(game)

    def latest_rating(self):
        if len(self.rating_history) == 0:
            return 0
        else:
            return self.rating_history[-1].rating

    def get_rating(self, date):
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
                    rating = prev_r.rating + (r.rating - prev_r.rating) * (date - prev_r.date) / (r.date - prev_r.date)
                    return rating
        return self.rating_history[-1].rating

    def init_rating_history(self):
        if self.root: return
        min_date = min((g.date for g in self.games), default=0)
        root_date = min_date - 100
        self.add_game(Game(root_date, self, root_player))
        self.add_game(Game(root_date, root_player, self))
        dates = list(set(g.date for g in self.games))
        dates.sort()
        self.games.sort(key=lambda g: g.date)

        self.rating_history = [RatingDatum(d, 0) for d in dates]
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
        elo_wsq = 300
        wsq = elo_wsq * 0.001

        num_points = len(self.rating_history)
        H = np.eye(num_points) * -0.001
        g = np.zeros(num_points)
        for (i,r) in enumerate(self.rating_history):
            my_gamma = r_to_gamma(r.rating)
            dprint("my gamma {} -> {}".format(r.rating, my_gamma))
            g[i] += len(r.wins)                              # Bradley-Terry
            for w in r.wins + r.losses:
                dprint("{} @ {} gamma {} -> ".format(w, r.date, w.get_rating(r.date)), end="")
                their_gamma = r_to_gamma(w.get_rating(r.date))
                dprint("{}".format(their_gamma))
                factor = 1. / (my_gamma + their_gamma)
                g[i] -= my_gamma * factor                    # Bradley-Terry
                H[i,i] -= my_gamma * their_gamma * factor**2 # Bradley-Terry
            if i > 0:
                dr = r.rating - self.rating_history[i-1].rating
                dt = r.date - self.rating_history[i-1].date
                sigmasq_recip = 1./(dt * wsq)
                dprint("dr {} dt {} sigmasq_recip {}".format(dr, dt, sigmasq_recip))
                g[i] -= dr * sigmasq_recip                   # Wiener
                H[i,i] -= sigmasq_recip                      # Wiener
                if i >= 1:
                    g[i-1] += dr * sigmasq_recip             # Wiener
                    H[i-1,i-1] -= sigmasq_recip              # Wiener
                H[i-1,i] += sigmasq_recip                    # Wiener
                H[i,i-1] += sigmasq_recip                    # Wiener
        return (H, g)

    # Return magnitude of changes
    def iterate_whr(self):
        if self.root: return 0.0
        dprint("\niterate_whr {} {}".format(self.handle, self.rating_history))

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

        dprint("g", g)
        dprint("H", H)
        dprint("d", d)
        dprint("b", b)
        dprint("a", a)
        dprint("x", x)
        # dprint("xn", xn)
        dprint("y", y)

        dprint("new ratings {} {}".format(self.handle, self.rating_history))
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
    def __init__(self, date, winner, loser):
        self.date = date
        self.winner = winner
        self.loser = loser

    def __repr__(self):
        return "{:02}: {} > {}".format(self.date, self.winner, self.loser)

player_db = {}                  # indexed by handle
games = []

def get_player(name, handle, is_root=False):
    if handle in player_db:
        player = player_db[handle]
        assert(player.name == name)
        return player
    else:
        player = Player(name, handle, is_root)
        player_db[handle] = player
        return player

root_player = get_player("[root]", "[root]", is_root=True)

# A full cycle name is something like "AYD League B, March 2014".
# Get just the date part
def cycle_to_date(s):
    return s.split(", ")[-1]

def parse_seasons(out_fname):
    # There are three cycles (e.g., "January 2014", "February 2014", "March 2014") in a season (e.g., 8)
    total_num_cycles = 0            # number of cycles in all previous seasons combined

    seasons = range(8, 22)
    for season in seasons:
        fn = "seasons/season_{:02d}.html".format(season)
        soup = BeautifulSoup(open(fn), "lxml")

        # First find the names of the cycles
        season_cycles = [] # Names of cycles within this season, in chronological order

        cycle_tags = soup.find_all(is_cycle_name)
        for cycle_tag in cycle_tags:
            date = cycle_to_date(cycle_tag.contents[0])
            if date not in season_cycles:
                season_cycles.append(date)
        season_cycles.reverse()

        # Now find the crosstables
        crosstables = soup.find_all("table", id="pointsTable")
        for crosstable in crosstables:
            # Find the name and date of this table
            crosstable_name = crosstable.find_parent("table").previous_sibling
            while type(crosstable_name) is NavigableString or crosstable_name.name != "h3":
                crosstable_name = crosstable_name.previous_sibling
            crosstable_date = cycle_to_date(crosstable_name.contents[0])
            global_cycle = total_num_cycles + season_cycles.index(crosstable_date)

            # Construct the list of players, in order
            crosstable_players = []
            trs = crosstable.find_all("tr")
            for tr in trs:
                tds = tr.find_all("td")
                if len(tds) > 0:
                    name = tds[1].nobr.a.contents[0]
                    handle = tds[2].contents[0]
                    crosstable_players.append(get_player(name,handle))

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
                            game = Game(global_cycle, winner, loser)
                            games.append(game)
                            winner.add_game(game)
                            loser.add_game(game)
                    row_player_idx += 1

        # Add an extra month between seasons
        total_num_cycles += len(season_cycles) + 1

    with open(out_fname, "w") as out_file:
        for g in games:
            print('"{}","{}","{}","{}","{}"'.format(g.date,
                                                    g.winner.name,
                                                    g.winner.handle,
                                                    g.loser.name,
                                                    g.loser.handle),
                  file=out_file)

# As produced by analyze_seasons()
def read_games_file(fname):
    player_db.clear()
    games.clear()
    with open(fname) as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            (date, winner_name, winner_handle, loser_name, loser_handle) = row
            date = int(date)
            winner = get_player(winner_name, winner_handle)
            loser = get_player(loser_name, loser_handle)
            game = Game(date, winner, loser)
            winner.add_game(game)
            loser.add_game(game)

def init_whr():
    for p in player_db.values():
        p.init_rating_history()

def iterate_whr():
    sum_xsq = 0
    for p in player_db.values():
        sum_xsq += p.iterate_whr() * 2
    return math.sqrt(sum_xsq)

def run_whr():
    init_whr()
    for i in range(1000):
        # print("ITERATION {}".format(i))
        change = iterate_whr()
        avg_change = change / len(player_db) # maybe should be avg change per rating point?
        # print("avg change", avg_change)
        if avg_change < 0.01:
            print("{} iterations".format(i+1))
            break
    for p in player_db.values():
        p.compute_stds()

def print_ratings():
    for p in sorted(player_db.values(), key=lambda p: p.latest_rating(), reverse=True):
        if len(p.rating_history) > 0:
            print("{:<10} {:>5} ± {:.2f}: {}".format(p.handle,
                                                     rating_to_rank(p.latest_rating()),
                                                     p.rating_history[-1].std,
                                                     p.rating_history[1:]))

def save_rating_history(fname):
    with open(fname, "w") as f:
        for p in sorted(player_db.values(), key=lambda p: p.latest_rating(), reverse=True):
              p.write_rating_history(f)

def load_rating_history(fname):
    with open(fname) as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            (name, handle) = row[:2]
            p = get_player(name, handle)
            # print(p)
            p.read_rating_history(row[2:])

if args.parse_seasons:
    parse_seasons(args.games_file)
else:
    read_games_file(args.games_file)

if args.analyze_games:
    run_whr()
    save_rating_history(args.ratings_file)

if args.load_ratings:
    load_rating_history(args.ratings_file)

if args.print_ratings:
    print_ratings()
