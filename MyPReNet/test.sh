#! /bin/bash
#rm gpu.txt
#sh gpu.sh |  tee gpu.txt

j=0;
while [ $j -eq 0 ]
do 
sleep 2s
echo  "\n" 
i=0; 
for line in `nvidia-smi | grep -Eo '\ \ [0-9]{1,5}MiB' |grep -Eo '[0-9]{1,5}' `  #`cat gpu.txt`
do
     if [ $i -gt 4 ]
     then
       break
     fi

     echo $line;
     #echo $i;

if [ $line -lt 5600 ] 
 then 
j=`expr $j + 1`;
#echo $j
#python test_PReNet.py --logdir logs/RainTrainH/PReNet_mobile2_Atte_T6/ --save_path results/Rain100H_result_PReNet_mobile2_Atte_T6_test15/ --data_path datasets/test/Rain100H/rain0/ --recurrent_iter 6 --use_GPU True --gpu_id $i

#python train_PReNet.py  --save_path logs/RainTrainH/PReNet_mobile6_Atte_T6_relu6 --data_path datasets/train/RainTrainH/  --use_gpu True --gpu_id $i --batch_size 1 --recurrent_iter 6
#ython train_PReNetDcnV2.py  --save_path logs/RainTrainH/PReNet_DCNv2_T6 --data_path datasets/train/RainTrainH/  --use_gpu True --gpu_id $i --batch_size 4 --recurrent_iter 6
#python test.py
#python train_PReNet.py  --save_path logs/BladeTrain/PReNet_baseline --data_path datasets/train/BladeTrainL  --use_gpu True --gpu_id $i --batch_size 8 --recurrent_iter 4
#python train_PReNetNewLapLoss.py  --save_path logs/BladeTrain/PReNet_LapLoss --data_path datasets/train/BladeTrainL  --use_gpu True --gpu_id $i --batch_size 1 --recurrent_iter 4
python train_PReNetNewLapLoss.py  --save_path logs/BladeTrain/PReNet_LapLoss1e4 --data_path datasets/train/BladeTrainL  --use_gpu True --gpu_id $i --batch_size 16 --recurrent_iter 4
echo 'sucess';
fi
    
i=`expr $i + 1`;

done

done
