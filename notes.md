## Notes

For brute force algorithm, we can use some heuristics

* Don't need to check group participation of all n agents. Let original winner be A, target winner B. We can just check the n' agents who have B>A as preference for participation
* We can use MoV as an estimate for how large of a sizes we need to check. So if lower bound for Margin of Victory is k, we should only check group participation for >=k

Final target result

* Various voting rules
  * Borda, plurality, veto
  * Copeland, maximin, [Black](https://en.wikipedia.org/wiki/Black%27s_method)
* Various tiebreaking methods
  * Lexicographic tiebreaking
  * Voter wise tiebreaking
  * Singleton tiebreaking
* Various axiom satisfaction
  * Participation
  * (Group) Participation
  * Anonimity
  * Neutrality
  * Condorcet criterion
