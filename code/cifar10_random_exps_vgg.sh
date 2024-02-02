# Task bash files to be called by the overall bash file for all experiments of the specified model type.
# We provide all of the individual bash files per task and model to make individual experiments easy to call.

reset_cuda(){
    sleep 10    
}

DEVICE=$1
seed=$2
#############################################################
################ CIFAR10 RANDOM FORGETTING ##################
#############################################################

forget_perc=0.00165 # 100 samples
dataset=Cifar10
n_classes=10
weight_path=VGG16Cifar10.pth

# Run the Python script
CUDA_VISIBLE_DEVICES=$DEVICE python src/forget_random_main.py -net VGG16 -dataset $dataset -classes $n_classes -gpu -method lipschitz_forgetting -forget_perc $forget_perc -weight_path $weight_path -seed $seed
reset_cuda
# CUDA_VISIBLE_DEVICES=$DEVICE python src/forget_random_main.py -net VGG16 -dataset $dataset -classes $n_classes -gpu -method scrub -forget_perc $forget_perc -weight_path $weight_path -seed $seed
# reset_cuda
# CUDA_VISIBLE_DEVICES=$DEVICE python src/forget_random_main.py -net VGG16 -dataset $dataset -classes $n_classes -gpu -method baseline -forget_perc $forget_perc -weight_path $weight_path -seed $seed
# reset_cuda
# CUDA_VISIBLE_DEVICES=$DEVICE python src/forget_random_main.py -net VGG16 -dataset $dataset -classes $n_classes -gpu -method finetune -forget_perc $forget_perc -weight_path $weight_path -seed $seed
# reset_cuda
# CUDA_VISIBLE_DEVICES=$DEVICE python src/forget_random_main.py -net VGG16 -dataset $dataset -classes $n_classes -gpu -method amnesiac -forget_perc $forget_perc -weight_path $weight_path -seed $seed
# reset_cuda
# CUDA_VISIBLE_DEVICES=$DEVICE python src/forget_random_main.py -net VGG16 -dataset $dataset -classes $n_classes -gpu -method blindspot -forget_perc $forget_perc -weight_path $weight_path -seed $seed
# reset_cuda
# CUDA_VISIBLE_DEVICES=$DEVICE python src/forget_random_main.py -net VGG16 -dataset $dataset -classes $n_classes -gpu -method UNSIR -forget_perc $forget_perc -weight_path $weight_path -seed $seed
# reset_cuda
# CUDA_VISIBLE_DEVICES=$DEVICE python src/forget_random_main.py -net VGG16 -dataset $dataset -classes $n_classes -gpu -method retrain -forget_perc $forget_perc -weight_path $weight_path -seed $seed
# reset_cuda