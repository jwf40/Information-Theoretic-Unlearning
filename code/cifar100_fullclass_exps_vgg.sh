# Task bash files to be called by the overall bash file for all experiments of the specified model type.
# We provide all of the individual bash files per task and model to make individual experiments easy to call.

reset_cuda(){
    sleep 10    
}

DEVICE=$1
seed=$2
#############################################################
################ CIFAR100 ROCKET FORGETTING #################
#############################################################
declare -a StringArray=("rocket" "mushroom" "baby" "lamp" "sea") # classes to iterate over


dataset=Cifar100
n_classes=100
weight_path=VGG16Cifar100.pth 

for val in "${StringArray[@]}"; do
    forget_class=$val
    # Run the Python script
    
    # CUDA_VISIBLE_DEVICES=$DEVICE python src/forget_full_class_main.py -net VGG16 -dataset $dataset -classes $n_classes -gpu -method ssd_tuning -forget_class $forget_class -weight_path $weight_path -seed $seed
    # reset_cuda
    CUDA_VISIBLE_DEVICES=$DEVICE python src/forget_full_class_main.py -net VGG16 -dataset $dataset -classes $n_classes -gpu -method lipschitz_forgetting -forget_class $forget_class -weight_path $weight_path -seed $seed
    reset_cuda

    # CUDA_VISIBLE_DEVICES=$DEVICE python src/forget_full_class_main.py -net VGG16 -dataset $dataset -classes $n_classes -gpu -method gkt -forget_class $forget_class -weight_path $weight_path -seed $seed
    # reset_cuda
    
    # CUDA_VISIBLE_DEVICES=$DEVICE python src/forget_full_class_main.py -net VGG16 -dataset $dataset -classes $n_classes -gpu -method emmn -forget_class $forget_class -weight_path $weight_path -seed $seed
    # reset_cuda

    # CUDA_VISIBLE_DEVICES=$DEVICE python src/forget_full_class_main.py -net VGG16 -dataset $dataset -classes $n_classes -gpu -method scrub -forget_class $forget_class -weight_path $weight_path -seed $seed
    # reset_cuda
    # CUDA_VISIBLE_DEVICES=$DEVICE python src/forget_full_class_main.py -net VGG16 -dataset $dataset -classes $n_classes -gpu -method baseline -forget_class $forget_class -weight_path $weight_path -seed $seed
    # reset_cuda
    # CUDA_VISIBLE_DEVICES=$DEVICE python src/forget_full_class_main.py -net VGG16 -dataset $dataset -classes $n_classes -gpu -method finetune -forget_class $forget_class -weight_path $weight_path -seed $seed
    # reset_cuda
    # CUDA_VISIBLE_DEVICES=$DEVICE python src/forget_full_class_main.py -net VGG16 -dataset $dataset -classes $n_classes -gpu -method amnesiac -forget_class $forget_class -weight_path $weight_path -seed $seed
    # reset_cuda
    # CUDA_VISIBLE_DEVICES=$DEVICE python src/forget_full_class_main.py -net VGG16 -dataset $dataset -classes $n_classes -gpu -method blindspot -forget_class $forget_class -weight_path $weight_path -seed $seed
    # reset_cuda
    # CUDA_VISIBLE_DEVICES=$DEVICE python src/forget_full_class_main.py -net VGG16 -dataset $dataset -classes $n_classes -gpu -method UNSIR -forget_class $forget_class -weight_path $weight_path -seed $seed
    # reset_cuda
    # CUDA_VISIBLE_DEVICES=$DEVICE python src/forget_full_class_main.py -net VGG16 -dataset $dataset -classes $n_classes -gpu -method retrain -forget_class $forget_class -weight_path $weight_path -seed $seed
    # reset_cuda
done