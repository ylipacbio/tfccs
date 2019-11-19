set -vex -o pipefail
# This documented how I preprocess lambda digest dataset and train
# a multinomial model regression to CCS to genome mapping.
echo "Movie: m64002_190608_021007"
movie=m64002_190608_021007
db_dir=/pbi/dept/consensus/ccsqv/data/Mule/lambda/bam/m64002_190608_021007/V2Bmark_SMS_Beta2_LambdaDigestEagI_SP2p1_DA011306_FCR_pkmid500_3260307/
raw_csv=${db_dir}/fextract/chunk-0.fextract.csv
out_dir=/pbi/dept/secondary/siv/yli/jira/tak-59/multi-ccs2genome
stat_json=${out_dir}/${movie}.chunk-0.stat.json

chunk0=${out_dir}/chunk-0
chunk0_train=${chunk0}.train.npz
chunk0_test=${chunk0}.test.npz

echo "Step 1: Compute simple statistics of trainable columns from raw fextract.csv and save stat.json"
fextract2stat ${raw_csv} ${stat_json}

echo "Stpe 2: convert fextract.csv to npz, standardize features and extract Y arrays"
fextract2numpy ${raw_csv} ${chunk0} --stat-json ${stat_json} --num-train-rows 500000

echo "Stpe 3: create npz for another chunk e.g., chunk-9, put all rows to one file"
fextract2numpy ${db_dir}/fextract/chunk-9.fextract.csv ${out_dir}/chunk-9 --stat-json ${stat_json} --num-train-rows 10000000

function chunk0_500K_epochs1000() {
    model_dir=m64002_190608_021007.chunk0.500K.model 
    echo "Stpe 4: train multinomial model using input ${chunk0_train}"
    multinomial ${chunk0_train} ${model_dir} --name 'lambda-multinomial' --batch-size 32 --epochs 1000

    echo "Stpe 5: Check saved model"
    saved_model_cli show --dir ${model_dir} --tag_set serve --signature_def serving_default

    echo "Step 6: Evaluate on training dataset chunk-9"
    evalmodel ${model} chunk-9.train.npz  chunk-9.train.fextract.csv ${model}/chunk-9.eval.csv > ${model}/chunk-9.eval.log
}

function chunk9_1M_epochs200() {
    model_dir=m64002_190608_021007.chunk-9.1M.model
    echo "Step 6: Evaluate on training dataset"
    multinomial chunk-9.train.npz ${model_dir} --name 'lambda-multinomial' --batch-size 512 --epochs 200

    echo "Stpe 5: Check saved model"
    saved_model_cli show --dir ${model_dir} --tag_set serve --signature_def serving_default

    echo "Step 6: Evaluate on training dataset chunk-0"
    evalmodel ${model} chunk-0.train.npz  chunk-0.train.fextract.csv ${model}/chunk-0.eval.csv > ${model}/chunk-0.eval.log
}

chunk9_1M_epochs200
