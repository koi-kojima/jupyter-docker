set -eux

cd "~/"
curl -L -O https://www.dropbox.com/s/rip0mhsytvt14mh/pretrained_models_pd_mesh_net.zip
unzip pretrained_models_pd_mesh_net.zip
export DATASET_FOLDER=~/datasets
export UNZIP_FOLDER=~/pretrained_models_pd_mesh_net
cd $UNZIP_FOLDER && find . -type f -name '*.yml' -exec sed -i "s@DATASET_FOLDER@${DATASET_FOLDER}@g" {} +
export JOB_FOLDER=$UNZIP_FOLDER/coseg/aliens_mean_aggr
cd ~/meshnet/evaluation_utils/
python generate_dataset.py --f $JOB_FOLDER
