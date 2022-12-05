echo "######### Start training-script! #########"
echo '''   ____                  __ _ _   _            
  / __ \                / _(_) | | |           
 | |  | |_   _____ _ __| |_ _| |_| |_ ___ _ __ 
 | |  | \ \ / / _ \ `__|  _| | __| __/ _ \ `__|
 | |__| |\ V /  __/ |  | | | | |_| ||  __/ |   
  \____/  \_/ \___|_|  |_| |_|\__|\__\___|_|   
                                               
                                               '''
echo ''
echo ''
# wait for 2 seconds
sleep 2

if [ -f createDataset.py ]; then
    echo "⏳ --> Start creating dataset.py..."
    python3 createDataset.py
    echo "✅ --> Dataset createt!"
else
    echo "❌ --> Could not find createDataset.py!"
fi
echo ''
echo ''


#### Start training  GPT2_TrainingLoop.py ####
# wait for 2 seconds
sleep 2


if [ -f GPT2_TrainingLoop.py ]; then
    echo "⏳ --> Starting training GPT2_TrainingLoop.py..."
    echo "writing logfiles to log.txt"
    python3 GPT2_TrainingLoop.py >> log.txt
    echo "✅ --> Training GPT2_TrainingLoop.py finished!"
else
    echo "❌ --> Could not find GPT2_TrainingLoop.py!"
fi
echo ''
echo ''

#### Start training  GPT2_TrainingLoop.py ####
# wait for 2 seconds
sleep 2


if [ -f GPT2_TrainingLoopRap.py ]; then
    echo "⏳ --> Starting training GPT2_TrainingLoopRap.py..."
    echo "writing logfiles to log.txt"
    python3 GPT2_TrainingLoopRap.py >> log.txt
    echo "✅ --> Training GPT2_TrainingLoopRap.py finished!"
else
    echo "❌ --> Could not find GPT2_TrainingLoop.py!"
fi
echo ''
echo ''

#### Start training  GPT2_TrainingLoop.py ####
# wait for 2 seconds
sleep 2


if [ -f GPT2_TrainingLoopTop.py ]; then
    echo "⏳ --> Starting training GPT2_TrainingLoopTop.py..."
    echo "writing logfiles to log.txt"
    python3 GPT2_TrainingLoopTop.py >> log.txt
    echo "✅ --> Training GPT2_TrainingLoopTop.py finished!"
else
    echo "❌ --> Could not find GPT2_TrainingLoopTop.py!"
fi
echo ''
echo ''


echo "######### Training finished! #########"