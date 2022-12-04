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


if [ -f GPT2-Training.py ]; then
    echo "⏳ --> Starting training GPT2-Training.py..."
    echo "wrinting logfiles to log.txt"
    #python3 GPT2-Training.py >> log.txt
    echo "✅ --> Training GPT2-Training.py finished!"
else
    echo "❌ --> Could not find GPT2-Training.py!"
fi
echo ''
echo ''
# wait for 2 seconds
sleep 2


if [ -f GPT2_TrainingLoop.py ]; then
    echo "⏳ --> Starting training GPT2_TrainingLoop.py..."
    echo "wrinting logfiles to log.txt"
    #python3 GPT2_TrainingLoop.py >> log.txt
    echo "✅ --> Training GPT2_TrainingLoop.py finished!"
else
    echo "❌ --> Could not find GPT2_TrainingLoop.py!"
fi
echo ''
echo ''
echo "######### Training finished! #########"

