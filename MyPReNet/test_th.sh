#! /bin/bash
#rm gpu.txt
#sh gpu.sh |  tee gpu.txt

j=0;
while [ $j -eq 0 ]
do 
sleep 1s
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

if [ $line -lt 13000 ] 
 then 
j=`expr $j + 1`;
#echo $j

python train_ARNet_th.py  --save_path logs/BladeTrain/PReNet_concateTh_Batch8 --data_path datasets/train/BladeTrainL  --use_gpu True --gpu_id $i --batch_size 8 --recurrent_iter 4 
echo 'sucess';
fi
    
i=`expr $i + 1`;

done

done
