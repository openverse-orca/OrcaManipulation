# ssh -L 8010:localhost:8010 hwz@172.23.148.117

# python eval.py --task_config competition_warehouse.yaml --host localhost --port 8010 --prompt "First, the left arm pick up the fan box, and the right arm pick up the barcode scanner. Then, scan the code, place the fan box into the cardboard box, and put down the scanner. Finally, use both arms to close the box." --max_steps 500 --sleep

# python eval.py --rand_file rand_no_barcode.yaml --host localhost --port 8010 --prompt "First, the left arm pick up the fan box, and the right arm pick up the barcode scanner. Then, scan the code, place the fan box into the cardboard box, and put down the scanner. Finally, use both arms to close the box." --episodes 10  --max_steps 2000 --sleep --action_repeat 5
