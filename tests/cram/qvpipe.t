  $ DATADIR=$TESTDIR/../data/
  $ PYBIN=$TESTDIR/../../tfccs/
  $ IN_JSON=$DATADIR/qvpipe.json
  $ OUT_DIR=$TESTDIR/../out/qvpipe

Test1: --help
  $ qvpipe --help  1>&2 >/dev/null && echo $?
  0
  $ qvpipe gen --help  1>&2 >/dev/null && echo $?
  0
  $ qvpipe run --help  1>&2 >/dev/null && echo $?
  0

Test2: qvpipe gen
  $ OUT_JSON=$OUT_DIR/out.gen.json
  $ rm -f $OUT_JSON
  $ qvpipe gen ${OUT_JSON} 1>&2 >/dev/null && echo $?
  0
  $ ls $OUT_JSON >/dev/null && echo $?
  0

Test3: qvpipe run
  $ qvpipe run $IN_JSON 1>$OUT_DIR/run.1.log 2>$OUT_DIR/run.2.log && echo $?
  0
  $ ls tmp/model/base_map_probability.json >/dev/null && echo $?
  0
  $ ls tmp/model/features.stat.json >/dev/null && echo $?
  0
  $ ls tmp/model/features.order.json >/dev/null && echo $?
  0
  $ cat tmp/model/hg2.benchmark.sh | grep input=
  input=/pbi/dept/consensus/ccsqv/data/Mule/hg2/one_percent.hg2.Grch38.ccs2genome.tsv

  $ cat tmp/model/lambda.benchmark.sh | grep input=
  input=/pbi/dept/consensus/ccsqv/data/Mule/lambda/one_percent.lambda.arrowqv.ccs2genome.tsv
