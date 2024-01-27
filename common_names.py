def name_set(fn):
    names = set()
    for l in open(fn):
        names.add(l.split()[0])
    return names

ayd_names = name_set("ayd-report.txt")
eyd_names = name_set("eyd-report.txt")
both_names = ayd_names.intersection(eyd_names)
print(both_names)
