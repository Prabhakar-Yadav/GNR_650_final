#!/bin/bash
git clone https://github.com/Prabhakar-Yadav/GNR_650_final.git
cd GNR_650_final
bash setup.bash
python inference.py --test_dir <path_to_test_dir>
