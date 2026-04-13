# **A.2 Harmless instructions**

To construct the harmless datasets, we sample instructions from ALPACA (Taori et al., 2023). _D_ harmless[(train)] contains 128 instructions, and _D_ harmless[(val)][contains 32 instructions.] 

In §3, we evaluate over 100 instructions from ALPACA. 

Note that _D_ harmless[(train)][,] _[D]_ harmless[(val)][,][and][the][evaluation][dataset][are][all][pairwise][disjoint,][containing][no] overlapping instructions. 

```
Pickapartthesyntaxofthesentence"Johnlovestoswim".
```

```
Rewritethefollowingsentenceinthenegativeform:"Itwassunnyallday".
Describetwobenefitsofgoingtocollege
Puttogetheramemorialdayspeech
Whatistheaveragenumberofhoursofsleepapersonshouldget?
```

Figure 8: A random sample of instructions from _D_ harmless[(train)][.] 

18
