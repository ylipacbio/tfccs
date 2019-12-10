THISDIR:=$(dir $(abspath $(lastword $(MAKEFILE_LIST))))

fextract_csv=/pbi/dept/consensus/ccsqv/data/Mule/lambda/bam/m64002_190608_021007/V2Bmark_SMS_Beta2_LambdaDigestEagI_SP2p1_DA011306_FCR_pkmid500_3260307/fextract/chunk-0.fextract.csv
in_stat_json=/pbi/dept/secondary/siv/yli/jira/tak-59/multi-ccs2genome/m64002_190608_021007.chunk-0.stat.json

all: pylint
	echo noop
test: pylint cram utest
dump-tiny:
	python tfccs/fextract2numpy.py \
		${fextract_csv} \
		output \
		--stat-json ${in_stat_json} \
		--num-train-rows 10000 \
		--no-dump-remaining
dump-full:
	python tfccs/fextract2numpy.py \
		${fextract_csv} \
		--stat-json ${in_stat_json} \
		/pbi/dept/secondary/siv/testdata/ccsqv/Mule/lambda/tfccs/output
build:
	pip install --edit .
utest:
	pytest -v -s tests/unit/test_utils.py tests/unit/test_fextract2x.py
pylint:
	pylint --errors-only tfccs/*.py
pwd=${shell pwd}
cram:
	@TESTDIR='${pwd}/tests/cram' \
		/mnt/software/c/cram/0.7/bin/cram \
		${pwd}/tests/cram/*.t
format:
	@for f in `ls tfccs/*.py`; do \
		autopep8 --max-line-length=120 -i -r -j0 $$f; \
	done
push:
	git push origin HEAD:master
show:
	saved_model_cli show --dir /pbi/dept/secondary/siv/yli/jira/tak-97/naive-multinomial --tag_set serve --signature_def serving_default
run:
	saved_model_cli run --dir /pbi/dept/secondary/siv/yli/jira/tak-97/naive-multinomial --tag_set serve --signature_def serving_default --input_exprs="dense_input=[[0.03]]"
clean:
	@rm -rf tfccs.egg-info
	@find . -type f -name '*.pyc' -delete
	@find . -type d -name '__pycache__' -delete
