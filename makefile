all: test
	echo noop
test:
	python fextract2numpy.py /home/UNIXHOME/yli/for_the_people/zdz/chunk-0.fextract.v3.csv output
pylint:
	pylint --errors-only fextract2numpy.py
pwd=${shell pwd}
cram:
	@TESTDIR='${pwd}/tests/cram' \
		/mnt/software/c/cram/0.7/bin/cram \
		${pwd}/tests/cram/*.t
format:
	@autopep8 --max-line-length=120 -ir -j0 fextract2numpy.py
push:
	git push origin HEAD:master
