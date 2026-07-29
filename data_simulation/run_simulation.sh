
#dataset="pamap2"
dataset="realworldhar"
#dataset="sleepedf20"
for availability in complete medium low
do
  for stability in complete medium low
  do
    python simulate_incompleteness.py --dataset_name $dataset --availability $availability --stability $stability
  done
done
