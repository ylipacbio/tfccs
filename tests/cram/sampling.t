  $ DATADIR=$TESTDIR/../data/sampling
  $ PYBIN=$TESTDIR/../../tfccs/
  $ IN1=$DATADIR/error.fextract.csv
  $ IN2=$DATADIR/rand.fextract.csv
  $ OUT=$CRAMTMP/sampling.csv

Test1: sampling exists
  $ sampling --help  1>&2 >/dev/null && echo $?
  utils.py:INFO:args: ['--help']
  0

Test2: run a small data
  $ sampling ${IN1} ${IN2} 10 20 30 50 ${OUT}  1>&2 >/dev/null && echo $?
  * (glob)
  *Sampling 2 bases from a total of 3 qualified bases in file* (glob)
  *Sampling 3 bases from a total of 6 qualified bases in file* (glob)
  *Sampling 5 bases from a total of 6 qualified bases in file* (glob)
  0
  $ cat ${OUT} | wc -l | sed 's/^[ ]*//g'
  5
