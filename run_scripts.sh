cd fixed_source/square
python3 square.py | tee square.out
python3 postprocess.py
cd ../circle
python3 circle.py | tee circle.out
python3 postprocess.py
cd ../../
