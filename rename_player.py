import sqlite3
import sys

def run(oldname, newname):
    con = sqlite3.connect("whr.db")
    cur = con.cursor()
    cur.execute(f"UPDATE games SET winner='{newname}' WHERE winner='{oldname}'")
    cur.execute(f"UPDATE games SET loser='{newname}' WHERE loser='{oldname}'")
    cur.execute(f"UPDATE ratings SET player='{newname}' WHERE player='{oldname}'")
    cur.execute(f"UPDATE yd_ratings SET player='{newname}' WHERE player='{oldname}'")
    con.commit()

run(sys.argv[1], sys.argv[2])    
