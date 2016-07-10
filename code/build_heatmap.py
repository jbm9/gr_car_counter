#!/usr/bin/env python

# This pulls in all the carlog_* files in the current directory and
# makes a heatmap of the data.  It's relatively janky and serves
# precisely my little needs.  It may be more of a distraction than a
# help from anything you want to do with it.  Sorry about that.
#
# Copyright (c) 2016 Josh Myer <josh@joshisanerd.com>
# License: cc0 (I barely want my name on this, much less its hacked children)
#

import numpy as np
import datetime
import json

import os

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt


long_hits = []

for p in os.listdir("."):
    if not p.startswith("carlog_"):
        continue

    f = file(p)
    g = f.read()
    f.close()

    rows = filter(None, [map(float, l.split()) for l in g.split("\n")])
    
    long_hits.extend(rows)


fig, ax = plt.subplots(figsize=(20,10))

nv = 5 # vertical bins per timeslice
vertical_resolution = 4 # seconds per bin
BUCKET_SIZE = 900 # 15 minute buckets

tses = [ t for t,l in long_hits ]


bucket_min = min(tses) / BUCKET_SIZE
bucket_max = max(tses) / BUCKET_SIZE

# Force day-alignment
tzoffset = (-7+3) * 3600  # I want 9PM at the left for my purposes
bucket_min = (min(tses) - (min(tses)+tzoffset)%86400)/ BUCKET_SIZE
bucket_max = (max(tses) + (86400-((max(tses)+tzoffset)%86400))) / BUCKET_SIZE



bucket_span = (bucket_max - bucket_min)
n_days = np.ceil((bucket_max-bucket_min) / (86400.0/BUCKET_SIZE))
day_span = 86400/BUCKET_SIZE

by_slice = np.zeros(shape=(nv*n_days, day_span))



for ts, l in long_hits:    
    raw_bucket = int(ts/BUCKET_SIZE - bucket_min)
    bucket = raw_bucket % day_span
    dayno = raw_bucket / day_span
    
    nsecs = int(np.round(l))

    if nsecs < 1:
        continue

    if nsecs > 20:
        print "skip: %d" % nsecs
        continue

    nsecs /= vertical_resolution

    if nsecs > nv-1:
        nsecs = nv-1

    vbucket = np.floor(nv * dayno + (nv-nsecs-1))

    try:
        by_slice[vbucket,bucket] += 1 # np.log(2+nsecs)
    except:
        print t, vbucket, bucket, dayno, raw_bucket, nsecs
        raise
        
x_lims = map(lambda t: matplotlib.dates.date2num(datetime.datetime.fromtimestamp(t)), [bucket_min*BUCKET_SIZE, bucket_max*BUCKET_SIZE])
y_lims = [0, nv]

ax.imshow(by_slice, interpolation="none", aspect="auto") #, cmap=cm.gray_r)#, cmap=cm.prism_r)

for x in range(0, day_span, 3600/BUCKET_SIZE):
    lwd = 0.5
    plt.axvline(x-0.5, color="grey", linewidth=lwd)
    
for i in range(int(n_days)):
    plt.axhline(nv*(i+1)+0.5, color="black", linewidth = 3)
    
plt.ylim(nv*n_days, 0)
plt.title("Number of cars, by pass time")

plt.savefig("img/heatmap_pass_times_grc.png")
