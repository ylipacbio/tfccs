all: test
	echo noop
dump-tiny:
	python tfccs/fextract2numpy.py \
		/home/UNIXHOME/yli/for_the_people/zdz/chunk-0.fextract.v3.csv \
		output \
		--num-train-rows 10000 \
		--no-dump-remaining
dump-full:
	python tfccs/fextract2numpy.py \
		/home/UNIXHOME/yli/for_the_people/zdz/chunk-0.fextract.v3.csv \
		/pbi/dept/secondary/siv/testdata/ccsqv/Mule/lambda/tfccs/output
pylint:
	pylint --errors-only tfccs/*.py
pwd=${shell pwd}
cram:
	@TESTDIR='${pwd}/tests/cram' \
		/mnt/software/c/cram/0.7/bin/cram \
		${pwd}/tests/cram/*.t
format:
	@autopep8 --max-line-length=120 -ir -j0 tfccs/*.py
push:
	git push origin HEAD:master
